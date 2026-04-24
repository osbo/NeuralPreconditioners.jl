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

function _apply_diagonal_neumann(A::SparseMatrixCSC, d::AbstractVector, v::AbstractVector;
                                 k::Int, alpha::Float32)
    y = d .* v
    for _ in 1:k
        y = y .+ alpha .* (d .* (v .- A * y))
    end
    return y
end

function _make_gnn_apply_fn(A::SparseMatrixCSC, graph::SparseGraph, params;
                            precond_mode::Symbol=:diag,
                            neumann_k::Int=1)
    c, α = gnn_predict_with_relaxation(graph, params)
    d = Float32.(graph.diag_inv) .* c
    if precond_mode === :diag
        apply_M = v -> d .* v
    elseif precond_mode === :neumann
        k = max(0, neumann_k)
        apply_M = v -> _apply_diagonal_neumann(A, d, v; k=k, alpha=α)
    else
        error("Unsupported precond_mode=$(precond_mode). Use :diag or :neumann.")
    end
    return apply_M, d, c, α
end

function _sai_cosine_loss_general(A::SparseMatrixCSC, rhs_batch::AbstractMatrix, apply_M::Function)
    m = size(rhs_batch, 2)
    total = sum(1:m) do j
        v = rhs_batch[:, j]
        Av = A * v
        Mv = apply_M(Av)
        num = dot(Mv, v)
        denom = sqrt(sum(abs2, Mv) * sum(abs2, v) + 1e-12)
        1f0 - num / denom
    end
    return Float32(total / m)
end

function _residual_loss_general(A::SparseMatrixCSC, rhs_batch::AbstractMatrix, apply_M::Function)
    m = size(rhs_batch, 2)
    total = sum(1:m) do j
        b = @view rhs_batch[:, j]
        r = A * apply_M(b) - b
        sum(abs2, r) / (sum(abs2, b) + 1e-12)
    end
    return Float32(total / m)
end

function _gnn_loss(A, graph, rhs, params, loss_type::Symbol, anchor_weight::Float32;
                   precond_mode::Symbol=:diag, neumann_k::Int=1)
    apply_M, d, c, α = _make_gnn_apply_fn(A, graph, params;
                                           precond_mode=precond_mode,
                                           neumann_k=neumann_k)
    c_reg = mean(abs2, log.(c))
    base = if loss_type === :sai_cosine
        _sai_cosine_loss_general(A, rhs, apply_M)
    elseif loss_type === :frobenius
        precond_mode === :diag || error(":frobenius loss supports precond_mode=:diag only")
        frobenius_loss(A, d)
    elseif loss_type === :jacobi_relative
        _residual_loss_general(A, rhs, apply_M) /
            (_residual_loss_general(A, rhs, v -> Float32.(graph.diag_inv) .* v) + 1f-12)
    else  # :residual
        _residual_loss_general(A, rhs, apply_M)
    end
    α_reg = (precond_mode === :neumann) ? 1f-3 * (α - 1f0)^2 : 0f0
    return Float32(base) + anchor_weight * c_reg + α_reg
end

"""Mean ‖A(d⊙b) - b‖ / ‖b‖ over columns of `rhs` (diagonal preconditioner `d`)."""
function _mean_rel_residual_ma(A::SparseMatrixCSC, rhs::AbstractMatrix, d::AbstractVector)
    m = size(rhs, 2)
    m == 0 && return 0f0
    s = sum(1:m) do j
        b = @view rhs[:, j]
        r = A * (d .* b) - b
        sqrt(sum(abs2, r) / (sum(abs2, b) + 1f-12))
    end
    return Float32(s / m)
end

"""Mean ‖A(Mb) - b‖ / ‖b‖ over columns of `rhs` for a generic preconditioner apply `M`."""
function _mean_rel_residual_general(A::SparseMatrixCSC, rhs::AbstractMatrix, apply_M::Function)
    m = size(rhs, 2)
    m == 0 && return 0f0
    s = sum(1:m) do j
        b = @view rhs[:, j]
        r = A * apply_M(b) - b
        sqrt(sum(abs2, r) / (sum(abs2, b) + 1f-12))
    end
    return Float32(s / m)
end

function _gnn_loss_base(A, graph, rhs, d::AbstractVector, loss_type::Symbol;
                        precond_mode::Symbol=:diag, neumann_k::Int=1, alpha::Float32=1f0)
    apply_M = precond_mode === :diag ?
        (v -> d .* v) :
        (v -> _apply_diagonal_neumann(A, d, v; k=max(0, neumann_k), alpha=alpha))
    if loss_type === :sai_cosine
        _sai_cosine_loss_general(A, rhs, apply_M)
    elseif loss_type === :frobenius
        precond_mode === :diag || error(":frobenius loss supports precond_mode=:diag only")
        frobenius_loss(A, d)
    elseif loss_type === :jacobi_relative
        _residual_loss_general(A, rhs, apply_M) /
            (_residual_loss_general(A, rhs, v -> Float32.(graph.diag_inv) .* v) + 1f-12)
    else
        _residual_loss_general(A, rhs, apply_M)
    end
end

function _recursive_grad_sqnorm(g)::Float64
    g === nothing && return 0.0
    if g isa AbstractArray{<:Number}
        return sum(abs2, g; init=0.0)
    elseif g isa Tuple || g isa NamedTuple
        return sum(_recursive_grad_sqnorm(v) for v in values(g); init=0.0)
    else
        return 0.0
    end
end

"""
    print_gnn_training_probe(A, graph, params, rhs, loss_type, anchor_weight; io=stdout)

Print a one-off diagnostic block: loss decomposition (base vs log-c anchor), statistics of
learned scale factors `c`, SAI-style mean cosine alignment, and mean relative residual
``‖A D b - b‖/‖b‖`` for learned `D=diag(d)` vs plain Jacobi `diag_inv`.
"""
function print_gnn_training_probe(A::SparseMatrixCSC, graph::SparseGraph, params, rhs::AbstractMatrix,
                                   loss_type::Symbol, anchor_weight::Float32;
                                   precond_mode::Symbol=:diag,
                                   neumann_k::Int=1,
                                   io::IO=stdout, grad_norm::Bool=false)
    apply_M, d, c, α = _make_gnn_apply_fn(A, graph, params;
                                           precond_mode=precond_mode,
                                           neumann_k=neumann_k)
    d_jac = Float32.(graph.diag_inv)
    reg = mean(abs2, log.(c))
    base = Float32(_gnn_loss_base(A, graph, rhs, d, loss_type;
                                  precond_mode=precond_mode,
                                  neumann_k=neumann_k,
                                  alpha=α))
    total = base + anchor_weight * reg

    # Mean cosine alignment for SAI loss (same probes as training rhs)
    m = size(rhs, 2)
    mean_cos = if loss_type === :sai_cosine && m > 0
        cos_sum = sum(1:m) do j
            v = rhs[:, j]
            Av = A * v
            Mv = apply_M(Av)
            denom = sqrt(sum(abs2, Mv) * sum(abs2, v) + 1e-12)
            dot(Mv, v) / denom
        end
        Float32(cos_sum / m)
    else
        Float32(NaN)
    end

    rel_learned = _mean_rel_residual_general(A, rhs, apply_M)
    rel_jac     = _mean_rel_residual_ma(A, rhs, d_jac)
    ratio = rel_jac > 0 ? rel_learned / rel_jac : Float32(NaN)

    q = Statistics.quantile(Float64.(c), (0.05, 0.5, 0.95))
    println(io, "    │ loss decomp: total=$(round(total, sigdigits=6)) = base($(loss_type))=$(round(base, sigdigits=6)) + " *
               "$(anchor_weight)×log(c)²=$(round(anchor_weight * reg, sigdigits=6))")
    if loss_type === :sai_cosine && !isnan(mean_cos)
        println(io, "    │ SAI probes: mean cos(M(Av),v)=$(round(mean_cos, sigdigits=5)) " *
                   "(1−cos≈$(round(1 - mean_cos, sigdigits=5)), match base=$(round(base, sigdigits=5)))")
    end
    println(io, "    │ c=exp(0.25·out): min=$(round(minimum(c), sigdigits=4))  max=$(round(maximum(c), sigdigits=4))  " *
               "mean=$(round(mean(c), sigdigits=4))  std=$(round(std(c), sigdigits=4))  " *
               "p5/p50/p95=$(round(q[1], sigdigits=3))/$(round(q[2], sigdigits=3))/$(round(q[3], sigdigits=3))")
    if precond_mode === :neumann
        println(io, "    │ learned Neumann: k=$(max(0, neumann_k))  alpha=$(round(α, sigdigits=4))")
    end
    println(io, "    │ probe ‖AMb−b‖/‖b‖: learned=$(round(rel_learned, sigdigits=5))  jacobi=$(round(rel_jac, sigdigits=5))  " *
               "ratio=$(round(ratio, sigdigits=5)) " * (ratio < 1 ? "(better than Jacobi on probes)" :
                   ratio > 1 ? "(worse than Jacobi on probes)" : ""))

    if grad_norm
        _, gs = Zygote.withgradient(params) do ps
            _gnn_loss(A, graph, rhs, ps, loss_type, anchor_weight;
                      precond_mode=precond_mode, neumann_k=neumann_k)
        end
        gnorm = sqrt(_recursive_grad_sqnorm(gs[1]))
        println(io, "    │ ‖∇params‖₂ (same probe) = $(round(gnorm, sigdigits=5))")
    end
    return nothing
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
- `precond_mode`   : `:diag` (learned diagonal) or `:neumann` (learned diagonal + k residual sweeps)
- `neumann_k`      : number of learned Neumann correction sweeps when `precond_mode=:neumann`
- `verbose`        : print every N epochs; 0 = silent (default 10)
- `diagnostics`    : 0 = none; 1 = after each verbose epoch, print probe (loss split, c stats,
                   rel. residual vs Jacobi on fresh RHS); 2 = same + ‖∇‖₂ (extra backward)
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
                                precond_mode::Symbol=:diag,
                                neumann_k::Int=1,
                                verbose::Int=10,
                                diagnostics::Int=0,
                                rng::AbstractRNG=Random.default_rng())
    opt       = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, params)

    for epoch in 1:n_epochs
        epoch_loss = 0.0
        last_A = nothing
        for A in training_matrices
            last_A = A
            graph = build_graph(A)
            N     = size(A, 1)
            rhs   = randn(rng, Float64, N, n_rhs)

            loss, grads = Zygote.withgradient(params) do ps
                _gnn_loss(A, graph, rhs, ps, loss_type, anchor_weight;
                          precond_mode=precond_mode, neumann_k=neumann_k)
            end

            opt_state, params = Optimisers.update!(opt_state, params, grads[1])
            epoch_loss += loss
        end

        if verbose > 0 && epoch % verbose == 0
            avg = epoch_loss / length(training_matrices)
            println("  Epoch $(lpad(epoch, 3)) | mean batch loss = $(round(avg, sigdigits=6))")
            if diagnostics ≥ 1 && last_A !== nothing
                g = build_graph(last_A)
                N = size(last_A, 1)
                rhs_p = randn(rng, Float64, N, n_rhs)
                println("    │ held-out probe on last epoch matrix: n=$N nnz=$(nnz(last_A))")
                print_gnn_training_probe(last_A, g, params, rhs_p, loss_type, anchor_weight;
                                         precond_mode=precond_mode,
                                         neumann_k=neumann_k,
                                         grad_norm=diagnostics ≥ 2)
            end
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

All other keyword arguments from the matrix-list dispatch apply (`diagnostics`, etc.).
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
                                precond_mode::Symbol=:diag,
                                neumann_k::Int=1,
                                verbose::Int=10,
                                diagnostics::Int=0,
                                rng::AbstractRNG=Random.default_rng())
    opt       = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, params)

    for epoch in 1:n_epochs
        matrices   = [sample_matrix(problem_class, rng) for _ in 1:n_samples_per_epoch]
        epoch_loss = 0.0
        last_A = nothing

        for A in matrices
            last_A = A
            graph = build_graph(A)
            N     = size(A, 1)
            rhs   = randn(rng, Float64, N, n_rhs)

            loss, grads = Zygote.withgradient(params) do ps
                _gnn_loss(A, graph, rhs, ps, loss_type, anchor_weight;
                          precond_mode=precond_mode, neumann_k=neumann_k)
            end

            opt_state, params = Optimisers.update!(opt_state, params, grads[1])
            epoch_loss += loss
        end

        if verbose > 0 && epoch % verbose == 0
            avg = epoch_loss / n_samples_per_epoch
            println("  Epoch $(lpad(epoch, 3)) | mean batch loss = $(round(avg, sigdigits=6))")
            if diagnostics ≥ 1 && last_A !== nothing
                g = build_graph(last_A)
                N = size(last_A, 1)
                rhs_p = randn(rng, Float64, N, n_rhs)
                println("    │ held-out probe on last epoch matrix: n=$N nnz=$(nnz(last_A))")
                print_gnn_training_probe(last_A, g, params, rhs_p, loss_type, anchor_weight;
                                         precond_mode=precond_mode,
                                         neumann_k=neumann_k,
                                         grad_norm=diagnostics ≥ 2)
            end
        end
    end

    return params
end

"""
    fine_tune!(params, A, cfg; kwargs...) -> params

Fine-tune a pre-trained GNN preconditioner on a specific operator A.

# Keywords
- `verbose`     : if > 0, print every `verbose` steps (default 0 = silent)
- `diagnostics` : same as `train_preconditioner!` (0 / 1 / 2) when used with `verbose > 0`
"""
function fine_tune!(params,
                    A::SparseMatrixCSC,
                    ::GNNConfig;
                    n_steps::Int=20,
                    lr::Float32=1f-4,
                    n_rhs::Int=8,
                    loss_type::Symbol=:sai_cosine,
                    anchor_weight::Float32=1f-2,
                    precond_mode::Symbol=:diag,
                    neumann_k::Int=1,
                    verbose::Int=0,
                    diagnostics::Int=0,
                    rng::AbstractRNG=Random.default_rng())
    opt       = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, params)
    graph     = build_graph(A)
    N         = size(A, 1)

    for step in 1:n_steps
        rhs = randn(rng, Float64, N, n_rhs)
        loss, grads = Zygote.withgradient(params) do ps
            _gnn_loss(A, graph, rhs, ps, loss_type, anchor_weight;
                      precond_mode=precond_mode, neumann_k=neumann_k)
        end
        opt_state, params = Optimisers.update!(opt_state, params, grads[1])

        if verbose > 0 && step % verbose == 0
            println("  fine_tune step $(lpad(step, 4)) | batch loss = $(round(loss, sigdigits=6))")
            if diagnostics ≥ 1
                rhs_p = randn(rng, Float64, N, n_rhs)
                println("    │ held-out probe (fresh RHS):")
                print_gnn_training_probe(A, graph, params, rhs_p, loss_type, anchor_weight;
                                         precond_mode=precond_mode,
                                         neumann_k=neumann_k,
                                         grad_norm=diagnostics ≥ 2)
            end
        end
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
