# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

# ── Loss functions ────────────────────────────────────────────────────────────

"""
    sai_cosine_loss(A, rhs_batch, d) -> Float32

Sparse Approximate Inverse (SAI) cosine-similarity loss:

    L = (1/m) Σⱼ [1 - cos(M(A bⱼ), bⱼ)]
      = (1/m) Σⱼ [1 - dot(d .* (A bⱼ), bⱼ) / (‖d .* (A bⱼ)‖ · ‖bⱼ‖)]

Trains the diagonal preconditioner M = diag(d) to approximate A⁻¹ by
maximising cosine alignment between M(Av) and v over random probes v.
Scale-invariant in d; no reference preconditioner required.

Optimal M satisfies M(Av) ∝ v for all v, i.e. M ∝ A⁻¹.
"""
function sai_cosine_loss(A::SparseMatrixCSC, rhs_batch::AbstractMatrix,
                          d::AbstractVector)
    m = size(rhs_batch, 2)
    total = sum(1:m) do j
        v  = rhs_batch[:, j]
        Av = A * v
        Mv = d .* Av
        num   = dot(Mv, v)
        denom = sqrt(sum(abs2, Mv) * sum(abs2, v) + 1e-12)
        1f0 - num / denom
    end
    return Float32(total / m)
end

"""
    residual_loss(A, rhs_batch, d) -> Float32

Residual-based preconditioner loss:
    L = (1/m) Σᵢ ‖A (d .* bᵢ) - bᵢ‖² / ‖bᵢ‖²
"""
function residual_loss(A::SparseMatrixCSC, rhs_batch::AbstractMatrix,
                       d::AbstractVector)
    m = size(rhs_batch, 2)
    total = sum(1:m) do j
        b = @view rhs_batch[:, j]
        r = A * (d .* b) - b
        sum(abs2, r) / (sum(abs2, b) + 1e-12)
    end
    return total / m
end

"""
    jacobi_relative_loss(A, rhs_batch, d, d_jac) -> Float32

Relative residual loss vs. Jacobi:
    L = residual_loss(A, rhs_batch, d) / residual_loss(A, rhs_batch, d_jac)

Values < 1 indicate improvement over plain Jacobi.
"""
function jacobi_relative_loss(A::SparseMatrixCSC, rhs_batch::AbstractMatrix,
                              d::AbstractVector, d_jac::AbstractVector)
    num = residual_loss(A, rhs_batch, d)
    den = residual_loss(A, rhs_batch, d_jac) + 1f-12
    return num / den
end

"""
    frobenius_loss(A, d) -> Float64

Exact Frobenius loss ‖diag(d)·A - I‖²_F for a diagonal preconditioner.
"""
function frobenius_loss(A::SparseMatrixCSC, d::AbstractVector)
    rows, cols, vals = findnz(A)
    return sum(eachindex(vals)) do k
        a = eltype(d)(vals[k])
        if rows[k] == cols[k]
            (d[cols[k]] * a - one(a))^2
        else
            (d[cols[k]] * a)^2
        end
    end
end

"""
    transformer_residual_loss(A, rhs_batch, params, cfg) -> Float32

Residual loss for the block-diagonal transformer preconditioner.
"""
function transformer_residual_loss(A::SparseMatrixCSC, rhs_batch::AbstractMatrix,
                                    params, cfg::TransformerConfig)
    p   = cfg.block_size
    n   = size(A, 1)
    K   = cld(n, p)
    A32 = SparseMatrixCSC{Float32}(A)

    blocks, _, _ = _extract_blocks(A, p)
    total = zero(Float32)
    m = size(rhs_batch, 2)

    for j in 1:m
        b = Float32.(rhs_batch[:, j])

        block_vecs = map(1:K) do k
            i_start = (k - 1) * p + 1
            i_end   = min(k * p, n)
            sz      = i_end - i_start + 1
            M_k     = _encode_block(blocks[k, :, :], params)
            M_k[1:sz, 1:sz] * b[i_start:i_end]
        end
        Mb = vcat(block_vecs...)

        r     = A32 * Mb .- b
        total += sum(abs2, r) / (sum(abs2, b) + 1f-12)
    end
    return total / m
end

# ── Shared gradient step ──────────────────────────────────────────────────────

function _gnn_loss(A, graph, rhs, params, loss_type::Symbol, anchor_weight::Float32)
    c = gnn_predict(graph, params)
    d = Float32.(graph.diag_inv) .* c
    c_reg = mean(abs2, log.(c))
    base = if loss_type === :sai_cosine
        sai_cosine_loss(A, rhs, d)
    elseif loss_type === :frobenius
        frobenius_loss(A, d)
    elseif loss_type === :jacobi_relative
        jacobi_relative_loss(A, rhs, d, graph.diag_inv)
    else  # :residual
        residual_loss(A, rhs, d)
    end
    return Float32(base) + anchor_weight * c_reg
end

# ── GNN training: matrix-list dispatch ───────────────────────────────────────

"""
    train_preconditioner!(params, training_matrices, cfg; kwargs...) -> params

Train the GNN diagonal preconditioner on a fixed collection of sparse matrices.

# Keyword arguments
- `n_epochs`       : training epochs (default 50)
- `lr`             : Adam learning rate (default 1f-3)
- `n_rhs`          : random RHS vectors per matrix per step (default 8)
- `loss_type`      : `:sai_cosine` | `:residual` | `:jacobi_relative` | `:frobenius`
- `anchor_weight`  : regularisation keeping c near 1 (default 1f-2)
- `verbose`        : print every N epochs; 0 = silent (default 10)
- `rng`            : random number generator
"""
function train_preconditioner!(params,
                                training_matrices::AbstractVector,
                                cfg::GNNConfig;
                                n_epochs::Int=50,
                                lr::Float32=1f-3,
                                n_rhs::Int=8,
                                loss_type::Symbol=:sai_cosine,
                                anchor_weight::Float32=1f-2,
                                verbose::Int=10,
                                rng::AbstractRNG=Random.default_rng())
    opt       = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, params)

    for epoch in 1:n_epochs
        epoch_loss = 0.0
        for A in training_matrices
            graph = build_graph(A)
            N     = size(A, 1)
            rhs   = randn(rng, Float64, N, n_rhs)

            loss, grads = Zygote.withgradient(params) do ps
                _gnn_loss(A, graph, rhs, ps, loss_type, anchor_weight)
            end

            opt_state, params = Optimisers.update!(opt_state, params, grads[1])
            epoch_loss += loss
        end

        if verbose > 0 && epoch % verbose == 0
            avg = epoch_loss / length(training_matrices)
            println("  Epoch $(lpad(epoch, 3)) | loss = $(round(avg, sigdigits=6))")
        end
    end

    return params
end

# ── GNN training: problem-class dispatch ─────────────────────────────────────

"""
    train_preconditioner!(params, class, cfg; kwargs...) -> params

Train the GNN preconditioner on instances sampled from a problem class.
Fresh matrices are drawn every epoch, providing implicit data augmentation.

# Additional keyword arguments (vs. matrix-list dispatch)
- `n_samples_per_epoch` : new matrices sampled per epoch (default 4)

All other keyword arguments from the matrix-list dispatch apply.
"""
function train_preconditioner!(params,
                                problem_class::AbstractProblemClass,
                                ::GNNConfig;
                                n_epochs::Int=50,
                                n_samples_per_epoch::Int=4,
                                lr::Float32=1f-3,
                                n_rhs::Int=8,
                                loss_type::Symbol=:sai_cosine,
                                anchor_weight::Float32=1f-2,
                                verbose::Int=10,
                                rng::AbstractRNG=Random.default_rng())
    opt       = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, params)

    for epoch in 1:n_epochs
        matrices   = [sample_matrix(problem_class, rng) for _ in 1:n_samples_per_epoch]
        epoch_loss = 0.0

        for A in matrices
            graph = build_graph(A)
            N     = size(A, 1)
            rhs   = randn(rng, Float64, N, n_rhs)

            loss, grads = Zygote.withgradient(params) do ps
                _gnn_loss(A, graph, rhs, ps, loss_type, anchor_weight)
            end

            opt_state, params = Optimisers.update!(opt_state, params, grads[1])
            epoch_loss += loss
        end

        if verbose > 0 && epoch % verbose == 0
            avg = epoch_loss / n_samples_per_epoch
            println("  Epoch $(lpad(epoch, 3)) | loss = $(round(avg, sigdigits=6))")
        end
    end

    return params
end

"""
    fine_tune!(params, A, cfg; kwargs...) -> params

Fine-tune a pre-trained GNN preconditioner on a specific operator A.
"""
function fine_tune!(params,
                    A::SparseMatrixCSC,
                    ::GNNConfig;
                    n_steps::Int=20,
                    lr::Float32=1f-4,
                    n_rhs::Int=8,
                    loss_type::Symbol=:sai_cosine,
                    anchor_weight::Float32=1f-2,
                    rng::AbstractRNG=Random.default_rng())
    opt       = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, params)
    graph     = build_graph(A)
    N         = size(A, 1)

    for _ in 1:n_steps
        rhs = randn(rng, Float64, N, n_rhs)
        _, grads = Zygote.withgradient(params) do ps
            _gnn_loss(A, graph, rhs, ps, loss_type, anchor_weight)
        end
        opt_state, params = Optimisers.update!(opt_state, params, grads[1])
    end

    return params
end

# ── Transformer training ──────────────────────────────────────────────────────

"""
    train_transformer!(params, training_matrices, cfg; kwargs...) -> params

Train the block-diagonal transformer preconditioner.
"""
function train_transformer!(params,
                             training_matrices::AbstractVector,
                             cfg::TransformerConfig;
                             n_epochs::Int=30,
                             lr::Float32=1f-3,
                             n_rhs::Int=4,
                             verbose::Int=10,
                             rng::AbstractRNG=Random.default_rng())
    opt       = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, params)

    for epoch in 1:n_epochs
        epoch_loss = 0.0
        for A in training_matrices
            N   = size(A, 1)
            rhs = randn(rng, Float64, N, n_rhs)

            loss, grads = Zygote.withgradient(params) do ps
                transformer_residual_loss(A, rhs, ps, cfg)
            end

            opt_state, params = Optimisers.update!(opt_state, params, grads[1])
            epoch_loss += loss
        end

        if verbose > 0 && epoch % verbose == 0
            avg = epoch_loss / length(training_matrices)
            println("  Epoch $(lpad(epoch, 3)) | loss = $(round(avg, sigdigits=6))")
        end
    end

    return params
end
