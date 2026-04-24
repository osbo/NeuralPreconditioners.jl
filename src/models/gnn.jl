# ─────────────────────────────────────────────────────────────────────────────
# Graph Neural Network preconditioner
#
# Architecture: multi-layer Graph Convolutional Network (GCN) that predicts
# per-node features used to construct a diagonal approximate inverse.
#
# Forward pass (one GCN layer):
#   H_agg = H · Â          (aggregate neighbour embeddings)
#   H_new = σ(W_agg H_agg + W_self H + b)
#
# Output: exp(0.25 * (W_out H + b_out)) → positive correction factors c
# Apply: M_θ b = (diag(A)^{-1} .* c) .* b  (Jacobi-corrected learned diagonal)
# ─────────────────────────────────────────────────────────────────────────────

"""
    GNNConfig

Hyperparameter record for the GNN preconditioner.
"""
Base.@kwdef struct GNNConfig
    node_dim   :: Int = 4      # input node-feature dimension
    hidden_dim :: Int = 64     # hidden embedding dimension
    n_layers   :: Int = 3      # number of GCN message-passing layers
    output_dim :: Int = 1      # outputs per node (1 for diagonal scaling)
    relax_max  :: Float32 = 1.5f0  # max learned relaxation for Neumann updates
end

# ── Parameter initialisation ──────────────────────────────────────────────────

function _glorot(rng, fan_out, fan_in)
    scale = sqrt(2.0f0 / (fan_in + fan_out))
    return randn(rng, Float32, fan_out, fan_in) .* scale
end

"""
    init_gnn_params(rng, cfg) -> NamedTuple

Randomly initialise all learnable parameters for the GNN.
Returns a nested NamedTuple compatible with Zygote and Optimisers.jl.

Parameter layout
────────────────
  params.embed  : initial linear embedding (node_dim → hidden_dim)
  params.layers : Tuple of GCN layer parameters (hidden_dim → hidden_dim)
  params.diag_head  : final linear layer (hidden_dim → output_dim)
  params.relax_head : pooled-embedding head for Neumann relaxation α
"""
function init_gnn_params(rng::AbstractRNG, cfg::GNNConfig)
    d_in, d_h, d_out = cfg.node_dim, cfg.hidden_dim, cfg.output_dim

    embed = (
        W = _glorot(rng, d_h, d_in),
        b = zeros(Float32, d_h),
    )

    layers = Tuple(
        (
            W_agg  = _glorot(rng, d_h, d_h),
            W_msg  = _glorot(rng, d_h, d_h),
            W_self = _glorot(rng, d_h, d_h),
            b      = zeros(Float32, d_h),
        )
        for _ in 1:cfg.n_layers
    )

    diag_head = (
        W = _glorot(rng, d_out, d_h),
        b = zeros(Float32, d_out),
    )

    relax_head = (
        w = vec(_glorot(rng, 1, d_h)),
        b = zeros(Float32, 1),
    )

    return (; embed, layers, diag_head, relax_head, relax_max=cfg.relax_max)
end

# ── Forward pass ──────────────────────────────────────────────────────────────

"""
    gnn_predict(graph, params) -> Vector{Float32}

Run the GNN on `graph` and return positive correction factors c
(length n, all positive), to be multiplied with `graph.diag_inv`.

Internally uses tanh for the embedding, relu for GCN layers, and a positive
exp-parametrisation centered at 1 for correction stability.
"""
function _gnn_forward_embeddings(graph::SparseGraph, params)
    H = graph.node_features           # d_node × n  (Float32)

    # Initial embedding: d_node → hidden_dim
    H = tanh.(params.embed.W * H .+ params.embed.b)  # hidden_dim × n

    # GCN message-passing layers
    for layer in params.layers
        # Two channels:
        #   - graph.A_hat: topology + magnitude (absolute normalized adjacency)
        #   - graph.A_msg: signed operator coupling (signed normalized adjacency)
        H_agg = H * graph.A_hat
        H_msg = H * graph.A_msg
        H = relu.(layer.W_agg * H_agg .+ layer.W_msg * H_msg .+ layer.W_self * H .+ layer.b)
    end
    return H
end

"""
    gnn_predict_with_relaxation(graph, params) -> (c, alpha)

Predict nodewise diagonal correction factors `c` and a scalar relaxation `alpha`
for Neumann-style residual correction.
"""
function gnn_predict_with_relaxation(graph::SparseGraph, params)
    H = _gnn_forward_embeddings(graph, params)

    # Diagonal correction: hidden_dim → 1, then squeeze to n-vector
    out = params.diag_head.W * H .+ params.diag_head.b
    c   = exp.(0.25f0 .* vec(out))                  # n, positive, starts near 1

    # Relaxation alpha from pooled embedding, bounded for stability.
    h_pool = vec(mean(H; dims=2))
    α_logit = dot(params.relax_head.w, h_pool) + params.relax_head.b[1]
    σ = 1f0 / (1f0 + exp(-Float32(α_logit)))
    α = params.relax_max * σ

    return c, α
end

function gnn_predict(graph::SparseGraph, params)
    c, _ = gnn_predict_with_relaxation(graph, params)
    return c
end

"""
    gnn_apply(graph, b, params) -> Vector

Apply the GNN preconditioner to vector b:
    M_θ b = (graph.diag_inv .* c) .* b
where c is the learned positive correction.
"""
function gnn_apply(graph::SparseGraph, b::AbstractVector, params)
    c = gnn_predict(graph, params)
    d = Float64.(graph.diag_inv) .* Float64.(c)
    return d .* b
end
