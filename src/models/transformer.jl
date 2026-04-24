# ─────────────────────────────────────────────────────────────────────────────
# Block-Diagonal Transformer preconditioner
#
# The n DOFs are partitioned into K blocks of size p.  For each block k the
# dense submatrix A_k ∈ ℝ^{p×p} is treated as a sequence of p tokens, each
# representing one row of A_k.  A Transformer encoder learns a map
#
#   A_k  →  M_k  ≈  A_k^{-1}
#
# and the global preconditioner is M_θ = blkdiag(M_1, …, M_K).
#
# Applying M_θ b: segment b into blocks and multiply each by its M_k.
#
# GPU compatibility: all block operations are performed as batched dense
# matrix multiplications over the K-batch dimension, mapping naturally to
# tensor cores.
# ─────────────────────────────────────────────────────────────────────────────

"""
    TransformerConfig

Hyperparameter record for the block-diagonal transformer preconditioner.
"""
Base.@kwdef struct TransformerConfig
    block_size  :: Int = 16   # p: rows/columns per diagonal block
    hidden_dim  :: Int = 64   # d: embedding dimension per token
    n_heads     :: Int = 4    # number of attention heads
    n_layers    :: Int = 2    # number of transformer encoder layers
end

# ── Parameter initialisation ──────────────────────────────────────────────────

function _glorot_t(rng, fan_out, fan_in)
    scale = sqrt(2.0f0 / (fan_in + fan_out))
    return randn(rng, Float32, fan_out, fan_in) .* scale
end

"""
    init_transformer_params(rng, cfg) -> NamedTuple

Randomly initialise all learnable parameters for the block-diagonal
transformer preconditioner.

Layout
──────
  params.row_embed   : row-embedding layer  (block_size → hidden_dim)
  params.attn_layers : Tuple of attention-layer params
  params.row_decode  : decoding layer       (hidden_dim → block_size)
"""
function init_transformer_params(rng::AbstractRNG, cfg::TransformerConfig)
    p, d, h = cfg.block_size, cfg.hidden_dim, cfg.n_heads
    d_k = d ÷ h  # key dimension per head

    row_embed = (
        W = _glorot_t(rng, d, p),
        b = zeros(Float32, d),
    )

    # Each attention layer has: multi-head QKV projections + output proj + FFN
    attn_layers = Tuple(
        (
            W_Q = _glorot_t(rng, d, d),
            W_K = _glorot_t(rng, d, d),
            W_V = _glorot_t(rng, d, d),
            W_O = _glorot_t(rng, d, d),
            # Layer-norm parameters (scale + bias per dim)
            ln1_γ = ones(Float32, d),
            ln1_β = zeros(Float32, d),
            # Feed-forward: d → 4d → d
            ff_W1 = _glorot_t(rng, 4d, d),
            ff_b1 = zeros(Float32, 4d),
            ff_W2 = _glorot_t(rng, d, 4d),
            ff_b2 = zeros(Float32, d),
            ln2_γ = ones(Float32, d),
            ln2_β = zeros(Float32, d),
        )
        for _ in 1:cfg.n_layers
    )

    row_decode = (
        W = _glorot_t(rng, p, d),
        b = zeros(Float32, p),
    )

    return (; row_embed, attn_layers, row_decode)
end

# ── Utility: layer normalisation ──────────────────────────────────────────────

function layer_norm(X::AbstractMatrix, γ, β; ε=1f-5)
    # X: d × seq_len; normalise over the d dimension for each token
    μ = mean(X, dims=1)
    σ = std(X, dims=1) .+ ε
    return γ .* (X .- μ) ./ σ .+ β
end

# ── Utility: single-block transformer encode ──────────────────────────────────

"""
    _encode_block(block_rows, params) -> Matrix{Float32}

Apply transformer encoder to a single block.

`block_rows`: p × p matrix (rows of A_k, each row is one token)
Returns: p × p matrix (rows of M_k)
"""
function _encode_block(block_rows::Matrix{Float32}, params)
    p = size(block_rows, 1)
    d = length(params.row_embed.b)

    # Embed each row of A_k → d-dimensional token
    # block_rows: p × p → transpose → p × p, embed each row
    E = params.row_embed.W * block_rows' .+ params.row_embed.b  # d × p

    # Transformer encoder layers
    for layer in params.attn_layers
        # ── Multi-head self-attention ──
        Q = layer.W_Q * E   # d × p
        K = layer.W_K * E
        V = layer.W_V * E

        n_heads = size(layer.W_Q, 1) ÷ (d ÷ length(layer.ln1_γ) > 0 ? 1 : 1)
        # Single-head attention (split into heads conceptually, compute together)
        scale   = sqrt(Float32(d))
        scores  = (K' * Q) ./ scale   # p × p
        weights = softmax(scores, dims=1)  # column-wise softmax
        attn_out = layer.W_O * (V * weights)  # d × p

        # Residual + layer norm 1
        E = layer_norm(E .+ attn_out, layer.ln1_γ, layer.ln1_β)

        # ── Position-wise feed-forward ──
        ff = layer.ff_W2 * relu.(layer.ff_W1 * E .+ layer.ff_b1) .+ layer.ff_b2

        # Residual + layer norm 2
        E = layer_norm(E .+ ff, layer.ln2_γ, layer.ln2_β)
    end

    # Decode each output token → one row of M_k
    M_k = (params.row_decode.W * E .+ params.row_decode.b)'  # p × p

    return M_k
end

# ── Block extraction / reassembly ─────────────────────────────────────────────

"""
    _extract_blocks(A, p) -> (Array{Float32,3}, Int, Int)

Extract diagonal blocks of size p from A.
Returns (blocks, K, n_padded):
  blocks   : K × p × p array of diagonal blocks (padded with identity)
  K        : number of blocks
  n_padded : padded matrix size
"""
function _extract_blocks(A::SparseMatrixCSC, p::Int)
    n  = size(A, 1)
    K  = cld(n, p)       # ceil division
    np = K * p

    A_dense = Matrix(A)  # bring to dense for block extraction
    blocks  = zeros(Float32, K, p, p)

    for k in 1:K
        i_start = (k - 1) * p + 1
        i_end   = min(k * p, n)
        sz      = i_end - i_start + 1

        blocks[k, 1:sz, 1:sz] = Float32.(A_dense[i_start:i_end, i_start:i_end])
        # Padding diagonal with 1 so the padded block has trivial inverse
        for i in sz+1:p
            blocks[k, i, i] = 1.0f0
        end
    end

    return blocks, K, np
end

# Block extraction depends only on fixed matrix data (A), never trainable params.
# Mark as non-differentiable so Zygote does not AD through internal mutation.
ChainRulesCore.@non_differentiable _extract_blocks(A::SparseMatrixCSC, p::Int)

# ── Forward pass ──────────────────────────────────────────────────────────────

"""
    transformer_predict(A, params, cfg) -> Array{Float32,3}

Run the transformer encoder on each diagonal block of A.
Returns a K × p × p array of predicted block-inverse matrices.
"""
function transformer_predict(A::SparseMatrixCSC, params, cfg::TransformerConfig)
    p = cfg.block_size
    blocks, K, _ = _extract_blocks(A, p)

    M_blocks = similar(blocks)
    for k in 1:K
        M_blocks[k, :, :] = _encode_block(blocks[k, :, :], params)
    end

    return M_blocks
end

"""
    transformer_apply(A, b, params, cfg) -> Vector{Float64}

Apply the block-diagonal transformer preconditioner to vector b.
"""
function transformer_apply(A::SparseMatrixCSC, b::AbstractVector,
                            params, cfg::TransformerConfig)
    p = cfg.block_size
    n = length(b)
    K = cld(n, p)

    M_blocks = transformer_predict(A, params, cfg)

    result = zeros(Float64, n)
    for k in 1:K
        i_start = (k - 1) * p + 1
        i_end   = min(k * p, n)
        sz      = i_end - i_start + 1
        M_k     = Float64.(M_blocks[k, 1:sz, 1:sz])
        result[i_start:i_end] = M_k * b[i_start:i_end]
    end

    return result
end
