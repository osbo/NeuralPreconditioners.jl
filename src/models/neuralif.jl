# ─────────────────────────────────────────────────────────────────────────────
# Neural Incomplete Factorization (NeuralIF) preconditioner
#
# Implements the architecture from Häusner et al. (arXiv:2305.16368):
#   "Neural incomplete factorization: learning preconditioners for CG"
#
# A GNN learns a sparse lower-triangular factor L with the same sparsity
# pattern as tril(A), trained to minimise the Hutchinson approximation of
# the Frobenius loss  E_w[ ‖LL'w − Aw‖₂ ].
#
# Architecture (two-pass blocks on lower-triangle edges + final MLP):
#   Each block: fwd (col→row) + bwd (row→col), each with edge MLP [18→32→32] + node MLP [40→32→8]
#   Final MLP: edge [18→32→1] — predicts raw L values
#   Diagonal activation: exp(z/2) → 1.0 at init; off-diagonal unconstrained
#
# Inference: L L' y = v  solved via two sparse triangular solves.
# ─────────────────────────────────────────────────────────────────────────────

# ── Shared initialisation helper ─────────────────────────────────────────────

function _glorot(rng, fan_out, fan_in)
    scale = sqrt(2.0f0 / (fan_in + fan_out))
    return randn(rng, Float32, fan_out, fan_in) .* scale
end

# ── Config ────────────────────────────────────────────────────────────────────

Base.@kwdef struct NeuralIFConfig
    n_node_features :: Int = 8
    d_edge          :: Int = 32
    n_layers        :: Int = 3
    hidden_size     :: Int = 32
    skip_connections :: Bool = true
end

# ── Graph representation ──────────────────────────────────────────────────────

"""
    NeuralIFGraph

Graph representation of a sparse SPD matrix for NeuralIF.

Stores the full edge set (lower + upper + diagonal) and precomputed
structures for differentiable scatter aggregation and triangular applies.
"""
struct NeuralIFGraph{
    VI  <: AbstractVector{<:Integer},   # index arrays (CPU: Vector{Int}, GPU: CuVector{Int32})
    VF  <: AbstractVector{Float32},     # Float32 feature vectors (CPU or GPU)
    MF  <: AbstractMatrix{Float32},     # Float32 feature matrices (CPU or GPU)
    SMD,                                # sparse Float64 matrix (CPU SparseMatrixCSC or CUSPARSE)
    VD  <: AbstractVector{<:Real}       # diagonal scaling (CPU or GPU dense/sparse)
}
    n             :: Int
    # Full edge index arrays — moveable to GPU (needed for NNlib.scatter aggregation)
    row_idx       :: VI
    col_idx       :: VI
    # Initial edge features: 2 × nnz_full  ([normalised a_ij; pos_enc])
    edge_init     :: MF
    # Per-edge inverse degree: 1/deg(row_idx[k]).
    # Replaces the old dense S_agg (nnz×n) matrix — O(nnz) instead of O(nnz×n).
    inv_deg_edge  :: VF
    # Lower-triangular sub-graph index arrays — moveable to GPU for NNlib.scatter
    lower_row     :: VI
    lower_col     :: VI
    lower_eidx    :: Vector{Int}        # always CPU: GPU fancy-indexing works with CPU idx
    is_diag            :: VF            # 1.0 for diagonal edges, 0.0 for strict lower
    # Per-edge inverse degree for two-pass lower-triangle scatter aggregation
    inv_deg_lower_row  :: VF            # 1/deg_row[lower_row[k]]: col→row forward sweep
    inv_deg_lower_col  :: VF            # 1/deg_col[lower_col[k]]: row→col backward sweep
    # Node features: 8 × n
    node_features      :: MF
    # Jacobi-scaled matrix — stored in graph precision (Float32 or Float64); moveable to CUSPARSE
    A_scaled      :: SMD
    # d_sqrt_inv[i] = 1/sqrt(A[i,i]); used at inference for D^{-1/2} scaling
    d_sqrt_inv    :: VD
end

"""
    build_neuralif_graph(A; precision=Float32) -> NeuralIFGraph

Construct a NeuralIFGraph from a sparse SPD matrix A.

`precision` controls the storage type of `A_scaled` and `d_sqrt_inv`
(the two fields that feed into preconditioner arithmetic). GNN features
are always Float32 regardless. Use `Float64` for maximum triangular-solve
accuracy at the cost of GPU memory and compute.
"""
function build_neuralif_graph(A::SparseMatrixCSC{T};
                               precision::Type{F}=Float32) where {T<:Real, F<:AbstractFloat}
    n   = size(A, 1)

    # ── Jacobi scaling: Â = D^{-1/2} A D^{-1/2} ──────────────────────────────
    # Always compute scaling in Float64 for numerical accuracy, then convert.
    d_sqrt_inv_f64 = 1.0 ./ sqrt.(abs.(Float64.(diag(A))) .+ 1e-12)
    ri0, ci0, nz0  = findnz(SparseMatrixCSC{Float64}(A))
    nz_s = nz0 .* d_sqrt_inv_f64[ri0] .* d_sqrt_inv_f64[ci0]
    A_s  = sparse(ri0, ci0, F.(nz_s), n, n)   # stored in target precision

    ri, ci, nz = findnz(A_s)
    nnz_full   = length(nz)

    # ── Edge features (on scaled matrix) ──────────────────────────────────────
    max_val   = maximum(abs.(nz))
    edge_norm = Float32.(nz) ./ (Float32(max_val) + 1f-12)
    pos_enc   = Float32[ri[k] > ci[k] ? -1f0 : (ri[k] == ci[k] ? 0f0 : 1f0)
                        for k in 1:nnz_full]
    edge_init = vcat(edge_norm', pos_enc')          # 2 × nnz_full

    # ── Node features (8, computed on scaled matrix) ──────────────────────────
    node_features = _build_neuralif_node_features(A_s, ri, ci, nz, n)

    # ── Per-edge inverse degree (replaces dense S_agg scatter matrix) ───────────
    # inv_deg_edge[k] = 1/deg(ri[k]): used to compute mean aggregation via scatter.
    # O(nnz_full) instead of O(nnz_full × n) — critical for large-problem GPU transfer.
    deg = zeros(Float32, n)
    for k in 1:nnz_full; deg[ri[k]] += 1f0; end
    inv_deg_edge = Float32[1f0 / max(deg[ri[k]], 1f0) for k in 1:nnz_full]

    # ── Lower triangular sub-graph ─────────────────────────────────────────────
    lower_mask = ri .>= ci
    lower_row  = ri[lower_mask]
    lower_col  = ci[lower_mask]
    lower_eidx = findall(lower_mask)
    is_diag    = Float32.(lower_row .== lower_col)

    # Inverse degrees for two-pass lower-triangle scatter (mean aggregation)
    deg_lower_row = zeros(Float32, n)
    deg_lower_col = zeros(Float32, n)
    for k in eachindex(lower_row)
        deg_lower_row[lower_row[k]] += 1f0
        deg_lower_col[lower_col[k]] += 1f0
    end
    inv_deg_lower_row = Float32[1f0 / max(deg_lower_row[lower_row[k]], 1f0) for k in eachindex(lower_row)]
    inv_deg_lower_col = Float32[1f0 / max(deg_lower_col[lower_col[k]], 1f0) for k in eachindex(lower_col)]

    d_sqrt_inv = F.(d_sqrt_inv_f64)

    return NeuralIFGraph(n, ri, ci, edge_init, inv_deg_edge,
                         lower_row, lower_col, lower_eidx, is_diag,
                         inv_deg_lower_row, inv_deg_lower_col,
                         node_features,
                         A_s,
                         d_sqrt_inv)
end

# ── Node feature construction (8 features per node) ───────────────────────────

function _build_neuralif_node_features(A, ri, ci, nz, n)
    diag_vals  = Vector{Float64}(diag(A))
    abs_A      = abs.(A)
    row_norms  = vec(sqrt.(sum(abs2.(A); dims=2)))
    col_norms  = vec(sqrt.(sum(abs2.(A); dims=1)))
    col_counts = Float64.(diff(A.colptr))   # nnz per column ≈ degree

    # Off-diagonal row sums for diagonal dominance/decay
    off_row_sum = zeros(Float64, n)
    off_row_max = zeros(Float64, n)
    for k in eachindex(nz)
        i, j = ri[k], ci[k]
        if i != j
            v = abs(nz[k])
            off_row_sum[i] += v
            off_row_max[i] = max(off_row_max[i], v)
        end
    end

    abs_diag = abs.(diag_vals)
    dom  = Float32.(abs_diag ./ (off_row_sum .+ 1e-12))   # diag dominance ratio
    dec  = Float32.(abs_diag ./ (off_row_max .+ 1e-12))   # diag decay ratio
    dom_feat = dom ./ (dom .+ 1f0)                          # mapped to (0,1)
    dec_feat = dec ./ (dec .+ 1f0)

    function _std_normalise(v::Vector{<:Real})
        μ, σ = mean(v), std(v)
        return Float32.((v .- μ) ./ (σ + 1e-6))
    end

    return vcat(
        _std_normalise(diag_vals)',
        _std_normalise(log.(row_norms  .+ 1e-12))',
        _std_normalise(log.(col_norms  .+ 1e-12))',
        _std_normalise(log.(col_counts .+ 1.0))',
        Float32.(sign.(diag_vals))',
        dom_feat',
        dec_feat',
        _std_normalise(log.(abs_diag  .+ 1e-12))',
    )   # 8 × n
end

# ── MLP helpers ───────────────────────────────────────────────────────────────

function _init_mlp(rng, sizes::Vector{Int})
    Tuple(
        (W = _glorot(rng, sizes[k+1], sizes[k]),
         b = zeros(Float32, sizes[k+1]))
        for k in 1:length(sizes)-1
    )
end

function _apply_mlp(x::AbstractMatrix, layers, activate_final::Bool=false)
    h = x
    for (i, layer) in enumerate(layers)
        h = layer.W * h .+ layer.b
        if i < length(layers) || activate_final
            h = relu.(h)
        end
    end
    return h
end

# ── Parameter initialisation ──────────────────────────────────────────────────

"""
    init_neuralif_params(rng, cfg) -> NamedTuple

Two-pass architecture: (n_layers-1) blocks of forward+backward message passing,
each block with separate fwd/bwd edge and node MLPs, plus a final edge prediction MLP.
Each block captures both the forward Cholesky sweep (col→row) and backward context
propagation (row→col), matching the paper's sequential Cholesky dependency structure.
"""
function init_neuralif_params(rng::AbstractRNG, cfg::NeuralIFConfig)
    d_n     = cfg.n_node_features   # e.g. 8
    d_e     = cfg.d_edge            # e.g. 32
    h       = cfg.hidden_size       # e.g. 32
    edge_in = 2 * d_n + 2           # [x_src; x_dst; e_init(2)] per lower-triangle edge

    n_blocks = max(cfg.n_layers - 1, 1)

    blocks = Tuple(
        (fwd = (edge_mlp = _init_mlp(rng, [edge_in, h, d_e]),
                node_mlp = _init_mlp(rng, [d_n + d_e, h, d_n])),
         bwd = (edge_mlp = _init_mlp(rng, [edge_in, h, d_e]),
                node_mlp = _init_mlp(rng, [d_n + d_e, h, d_n])))
        for _ in 1:n_blocks
    )
    final_mlp = _init_mlp(rng, [edge_in, h, 1])
    return (; blocks, final_mlp)
end

# ── Forward pass ──────────────────────────────────────────────────────────────

"""
    neuralif_forward(graph, params) -> Vector{Float32}

Run the two-pass NeuralIF GNN on `graph` and return lower-triangular L values
(length = nnz of tril(A), including diagonal).

Each block applies:
1. Forward sweep (col→row): aggregates along the lower-triangle graph, capturing
   the left-to-right dependency in Cholesky: L[i,j] depends on L[i,k] for k<j.
2. Backward sweep (row→col): propagates updated row context back to column nodes.

Diagonal entries are constrained positive via exp(z/2), which initialises to
1.0 at z=0 — correct for the unit-diagonal Jacobi-scaled system.
"""
function neuralif_forward(graph::NeuralIFGraph, params)
    n      = graph.n
    x      = graph.node_features                          # d_n × n
    lr     = graph.lower_row                               # nnz_lower
    lc     = graph.lower_col                               # nnz_lower
    e_init = graph.edge_init[:, graph.lower_eidx]         # 2 × nnz_lower
    inv_r  = graph.inv_deg_lower_row'                      # 1 × nnz_lower (for mean)
    inv_c  = graph.inv_deg_lower_col'                      # 1 × nnz_lower (for mean)

    for block in params.blocks
        # Forward sweep: col → row
        m_fwd   = _apply_mlp(vcat(x[:, lc], x[:, lr], e_init), block.fwd.edge_mlp, true)
        agg_fwd = scatter(+, m_fwd .* inv_r, lr; dstsize=(size(m_fwd, 1), n))
        x       = relu.(_apply_mlp(vcat(x, agg_fwd), block.fwd.node_mlp))

        # Backward sweep: row → col
        m_bwd   = _apply_mlp(vcat(x[:, lr], x[:, lc], e_init), block.bwd.edge_mlp, true)
        agg_bwd = scatter(+, m_bwd .* inv_c, lc; dstsize=(size(m_bwd, 1), n))
        x       = relu.(_apply_mlp(vcat(x, agg_bwd), block.bwd.node_mlp))
    end

    # Final edge prediction: one scalar per lower-triangle edge
    L_raw = vec(_apply_mlp(vcat(x[:, lr], x[:, lc], e_init), params.final_mlp))

    # exp(z/2): diagonal entries initialized to 1.0 at z=0, always positive.
    # Off-diagonal entries are unconstrained (positive or negative).
    L_vals = (1f0 .- graph.is_diag) .* L_raw .+
              graph.is_diag .* exp.(L_raw .* 0.5f0)
    return L_vals
end

# ── Triangular matrix-vector products (differentiable) ────────────────────────

"""
    _apply_L(L_vals, graph, v) -> Vector

Compute L * v via differentiable scatter-add (works on CPU and GPU):
    (L*v)[i] = Σ_{k: lower_row[k]=i} L_vals[k] * v[lower_col[k]]
"""
function _apply_L(L_vals::AbstractVector, graph::NeuralIFGraph, v::AbstractVector)
    scaled = L_vals .* v[graph.lower_col]
    return scatter(+, scaled, graph.lower_row; dstsize=(graph.n,))
end

"""
    _apply_Lt(L_vals, graph, v) -> Vector

Compute L' * v via differentiable scatter-add (works on CPU and GPU):
    (L'*v)[j] = Σ_{k: lower_col[k]=j} L_vals[k] * v[lower_row[k]]
"""
function _apply_Lt(L_vals::AbstractVector, graph::NeuralIFGraph, v::AbstractVector)
    scaled = L_vals .* v[graph.lower_row]
    return scatter(+, scaled, graph.lower_col; dstsize=(graph.n,))
end

# ── Hutchinson loss ───────────────────────────────────────────────────────────

"""
    hutchinson_loss(L_vals, graph, z) -> Float32

Unbiased estimator of ‖LL' − A‖_F via a single probe z ∼ N(0, I):
    loss = ‖LL'z − Az‖₂

Backpropagates through L_vals (GNN output) cleanly.
"""
function hutchinson_loss(L_vals::AbstractVector, graph::NeuralIFGraph,
                         z::AbstractVector)
    z32  = Float32.(z)
    y    = _apply_Lt(L_vals, graph, z32)             # L'z
    w    = _apply_L(L_vals, graph, y)                # L(L'z) = LL'z  (in scaled space)
    # A_scaled * z is constant w.r.t. params — exclude from AD tape entirely.
    # Float32.() is a no-op when A_scaled is already Float32; downcasts for Float64 graphs.
    Asz  = Zygote.ignore(() -> Float32.(graph.A_scaled * z32))
    r    = w .- Asz
    Asz_norm = sqrt(sum(abs2, Asz) + 1f-8)
    return sqrt(sum(abs2, r) + 1f-12) / Asz_norm
end

# ── Precompute sparse L and apply as preconditioner ───────────────────────────

"""
    neuralif_build_L(L_vals, graph) -> SparseMatrixCSC{F}

Assemble the sparse lower-triangular Cholesky factor L from predicted values.
Always returns a CPU sparse matrix in the graph's precision (Float32 or Float64).
Handles GPU `L_vals` by gathering to host first.
"""
function neuralif_build_L(L_vals::AbstractVector, graph::NeuralIFGraph)
    F    = eltype(graph.d_sqrt_inv)   # Float32 or Float64, matches graph precision
    vals = F.(Array(L_vals))          # GPU→CPU if needed, then convert
    return sparse(graph.lower_row, graph.lower_col, vals, graph.n, graph.n)
end

"""
    neuralif_apply(L, L_T, v) -> Vector{Float64}

Apply the NeuralIF preconditioner (LL')⁻¹ v via two triangular solves:
    solve L  z = v  (forward)
    solve L' y = z  (backward)
"""
function neuralif_apply(L_lower::LowerTriangular, L_T_upper::UpperTriangular,
                        v::AbstractVector)
    z = L_lower  \ Float64.(v)
    y = L_T_upper \ z
    return y
end
