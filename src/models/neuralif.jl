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
# Architecture (three GraphNet layers, all edges, positional encoding):
#   Layer 0 : edge 2  → 32, node [8+32] → 8
#   Layer 1 : edge 33 → 32, node [8+32] → 8   (skip: append original edge val)
#   Layer 2 : edge 33 → 1                      (final L values)
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
struct NeuralIFGraph
    n           :: Int
    # Full edge set (row_idx[k], col_idx[k]) for all nonzeros
    row_idx     :: Vector{Int}
    col_idx     :: Vector{Int}
    # Initial edge features: 2 × nnz_full  ([normalised a_ij; pos_enc])
    #   pos_enc: -1 = strictly lower, 0 = diagonal, +1 = strictly upper
    edge_init   :: Matrix{Float32}
    # Mean scatter: (d × nnz_full) edge_emb → (d × n) node messages
    #   m = edge_emb * S_agg   where S_agg is (nnz_full × n) precomputed
    S_agg       :: Matrix{Float32}
    # Lower-triangular sub-graph (i ≥ j), indices into full edge arrays
    lower_row   :: Vector{Int}
    lower_col   :: Vector{Int}
    lower_eidx  :: Vector{Int}       # which full-edge k is a lower-tri edge
    is_diag     :: Vector{Float32}   # 1.0 for diagonal edges, 0.0 for strict lower
    # Scatter matrices for triangular apply (n × nnz_lower sparse)
    S_row_L     :: SparseMatrixCSC{Float32}  # S_row_L[i,k]=1 if lower_row[k]=i  (for L*v)
    S_col_L     :: SparseMatrixCSC{Float32}  # S_col_L[j,k]=1 if lower_col[k]=j  (for L'*v)
    # Node features: 8 × n
    node_features :: Matrix{Float32}
    # Jacobi-scaled matrix (Float64), used for Hutchinson loss
    # A_scaled = D^{-1/2} A D^{-1/2}  (diagonal entries ≈ 1, off-diagonal ≈ O(1/n))
    A_scaled    :: SparseMatrixCSC{Float64}
    # Diagonal scaling: d_sqrt_inv[i] = 1/sqrt(A[i,i])
    # At inference: M^{-1}r = D^{-1/2} (L_s L_s^T)^{-1} D^{-1/2} r
    d_sqrt_inv  :: Vector{Float64}
end

"""
    build_neuralif_graph(A) -> NeuralIFGraph

Construct a NeuralIFGraph from a sparse SPD matrix A.
"""
function build_neuralif_graph(A::SparseMatrixCSC{T}) where {T<:Real}
    n   = size(A, 1)

    # ── Jacobi scaling: Â = D^{-1/2} A D^{-1/2} ──────────────────────────────
    # Scales the diagonal to 1.0, making L values O(1) across problem sizes.
    d_sqrt_inv = 1.0 ./ sqrt.(abs.(Float64.(diag(A))) .+ 1e-12)
    ri0, ci0, nz0 = findnz(SparseMatrixCSC{Float64}(A))
    nz_s = nz0 .* d_sqrt_inv[ri0] .* d_sqrt_inv[ci0]   # scale entries
    A_s  = sparse(ri0, ci0, nz_s, n, n)

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

    # ── Mean scatter matrix S_agg (nnz_full × n) ──────────────────────────────
    deg = zeros(Float32, n)
    for k in 1:nnz_full; deg[ri[k]] += 1f0; end
    S_agg = zeros(Float32, nnz_full, n)
    for k in 1:nnz_full; S_agg[k, ri[k]] = 1f0 / max(deg[ri[k]], 1f0); end

    # ── Lower triangular sub-graph ─────────────────────────────────────────────
    lower_mask = ri .>= ci
    lower_row  = ri[lower_mask]
    lower_col  = ci[lower_mask]
    lower_eidx = findall(lower_mask)
    is_diag    = Float32.(lower_row .== lower_col)
    nnz_lower  = length(lower_row)

    S_row_L = sparse(lower_row, 1:nnz_lower, ones(Float32, nnz_lower), n, nnz_lower)
    S_col_L = sparse(lower_col, 1:nnz_lower, ones(Float32, nnz_lower), n, nnz_lower)

    return NeuralIFGraph(n, ri, ci, edge_init, S_agg,
                         lower_row, lower_col, lower_eidx, is_diag,
                         S_row_L, S_col_L,
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
"""
function init_neuralif_params(rng::AbstractRNG, cfg::NeuralIFConfig)
    d_n = cfg.n_node_features
    d_e = cfg.d_edge
    h   = cfg.hidden_size

    layer_params = Tuple(
        let is_first = (l == 1),
            is_last  = (l == cfg.n_layers),
            d_e_in   = is_first ? 2 : (cfg.skip_connections ? d_e + 1 : d_e),
            d_e_out  = is_last  ? 1 : d_e
            (
                edge_mlp = _init_mlp(rng, [2*d_n + d_e_in, h, d_e_out]),
                node_mlp = is_last ? nothing :
                                     _init_mlp(rng, [d_n + d_e_out, h, d_n]),
            )
        end
        for l in 1:cfg.n_layers
    )

    return (; layers=layer_params)
end

# ── Forward pass ──────────────────────────────────────────────────────────────

"""
    neuralif_forward(graph, params) -> Vector{Float32}

Run the NeuralIF GNN on `graph` and return lower-triangular L values (length
= nnz of tril(A), including diagonal). Diagonal entries are constrained
positive via softplus.
"""
function neuralif_forward(graph::NeuralIFGraph, params)
    n = graph.n
    x = graph.node_features           # 8 × n  (Float32, constant)
    e = graph.edge_init                # 2 × nnz_full

    # Store original edge values for skip connections (1 × nnz_full)
    e_orig = e[1:1, :]                 # row view of normalised a_ij values

    for (l, lp) in enumerate(params.layers)
        is_last = (l == length(params.layers))

        # Skip connection: append original edge values to embedding
        if l > 1 && !isnothing(lp.node_mlp)
            e = vcat(e, e_orig)        # (d_e + 1) × nnz_full
        elseif l > 1
            # last layer also gets skip
            e = vcat(e, e_orig)
        end

        # Edge update: input is [x[row]; x[col]; e] per edge
        X_row = x[:, graph.row_idx]    # 8 × nnz_full
        X_col = x[:, graph.col_idx]    # 8 × nnz_full
        E_in  = vcat(X_row, X_col, e)  # (16 + d_e_in) × nnz_full
        e_new = _apply_mlp(E_in, lp.edge_mlp, !is_last)  # d_e_out × nnz_full

        if !is_last
            # Node aggregation (mean) and update
            m   = e_new * graph.S_agg  # d_e × n  (mean of incident edge embeddings)
            X_in = vcat(x, m)          # (8 + d_e) × n
            x   = relu.(_apply_mlp(X_in, lp.node_mlp))  # 8 × n
        end

        e = e_new
    end

    # Extract lower-triangular output values  (nnz_lower)
    L_raw = vec(e[1, graph.lower_eidx])

    # Diagonal: softplus for positivity; off-diagonal: unconstrained
    eps = 1f-3
    L_vals = (1f0 .- graph.is_diag) .* L_raw .+
              graph.is_diag .* (softplus.(L_raw) .+ eps)

    return L_vals
end

# ── Triangular matrix-vector products (differentiable) ────────────────────────

"""
    _apply_L(L_vals, graph, v) -> Vector

Compute L * v using scatter: (L*v)[i] = Σ_{k: lower_row[k]=i} L_vals[k] * v[lower_col[k]]
"""
function _apply_L(L_vals::AbstractVector, graph::NeuralIFGraph, v::AbstractVector)
    scaled = L_vals .* v[graph.lower_col]   # nnz_lower
    return vec(graph.S_row_L * scaled)       # n
end

"""
    _apply_Lt(L_vals, graph, v) -> Vector

Compute L' * v using scatter: (L'*v)[j] = Σ_{k: lower_col[k]=j} L_vals[k] * v[lower_row[k]]
"""
function _apply_Lt(L_vals::AbstractVector, graph::NeuralIFGraph, v::AbstractVector)
    scaled = L_vals .* v[graph.lower_row]   # nnz_lower
    return vec(graph.S_col_L * scaled)       # n
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
    Asz  = Float32.(graph.A_scaled * Float64.(z32))  # A_scaled * z  (constant wrt params)
    r    = w .- Asz
    Asz_norm = sqrt(sum(abs2, Asz) + 1f-8)
    return sqrt(sum(abs2, r) + 1f-12) / Asz_norm
end

# ── Precompute sparse L and apply as preconditioner ───────────────────────────

"""
    neuralif_build_L(L_vals, graph) -> SparseMatrixCSC{Float64}

Assemble the sparse lower-triangular Cholesky factor L from predicted values.
"""
function neuralif_build_L(L_vals::AbstractVector, graph::NeuralIFGraph)
    return sparse(graph.lower_row, graph.lower_col,
                  Float64.(L_vals), graph.n, graph.n)
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
