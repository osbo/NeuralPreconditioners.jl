# ─────────────────────────────────────────────────────────────────────────────
# Training utilities for the NeuralIF preconditioner
# ─────────────────────────────────────────────────────────────────────────────

# ── Helper ────────────────────────────────────────────────────────────────────

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

"""Append all numeric gradient arrays in `g` (depth-first) into `buf` (Float64)."""
function _accum_grad_vec!(buf::Vector{Float64}, g)
    g === nothing && return
    if g isa AbstractArray{<:Number}
        append!(buf, vec(Float64.(g)))
    elseif g isa Tuple || g isa NamedTuple
        for v in values(g)
            _accum_grad_vec!(buf, v)
        end
    end
    return nothing
end

function _neuralif_grad_flat(A::SparseMatrixCSC, graph::NeuralIFGraph, params,
                             rhs_batch::AbstractMatrix)
    _, gs = Zygote.withgradient(params) do ps
        _neuralif_step_loss(A, graph, ps, rhs_batch)
    end
    buf = Float64[]
    _accum_grad_vec!(buf, gs[1])
    return buf
end

# ── Per-step loss (used inside Zygote.withgradient) ──────────────────────────

function _neuralif_step_loss(A::SparseMatrixCSC, graph::NeuralIFGraph,
                              params, rhs_batch::AbstractMatrix)
    L_vals = neuralif_forward(graph, params)
    m      = size(rhs_batch, 2)
    total  = sum(1:m) do j
        hutchinson_loss(L_vals, graph, Float32.(rhs_batch[:, j]))
    end
    return Float32(total / m)
end

# ── Diagnostic probe ─────────────────────────────────────────────────────────

"""
    print_neuralif_probe(A, graph, params, rhs; io=stdout)

Print diagnostics: mean Hutchinson loss, L diagonal stats,
and mean relative residual ‖LL'b - Ab‖/‖Ab‖ vs Jacobi.
"""
function print_neuralif_probe(A::SparseMatrixCSC, graph::NeuralIFGraph,
                               params, rhs::AbstractMatrix; io::IO=stdout)
    L_vals = Float64.(neuralif_forward(graph, params))
    m      = size(rhs, 2)

    # Hutchinson losses
    mean_h_loss = mean(hutchinson_loss(Float32.(L_vals), graph,
                                       Float32.(rhs[:, j])) for j in 1:m)

    # Diagonal of L
    d_L = L_vals[graph.is_diag .== 1f0]

    # Mean ‖L_s L_s'b_s − A_s b_s‖ / ‖A_s b_s‖ in scaled space
    rel_res = mean(1:m) do j
        b_s  = Float32.(graph.d_sqrt_inv) .* Float32.(rhs[:, j])
        Asbs = Float32.(graph.A_scaled * Float64.(b_s))
        L32  = Float32.(L_vals)
        Ltbs = _apply_Lt(L32, graph, b_s)
        LLtbs = _apply_L(L32, graph, Ltbs)
        norm(Float64.(LLtbs) .- Float64.(Asbs)) / (norm(Asbs) + 1e-12)
    end

    # Jacobi relative residual on original A (for reference)
    d_jac = 1.0 ./ diag(A)
    rel_jac = mean(1:m) do j
        b = rhs[:, j]
        r = A * (d_jac .* b) - b
        norm(r) / (norm(b) + 1e-12)
    end

    println(io, "    │ Hutchinson loss: $(round(mean_h_loss, sigdigits=5))")
    println(io, "    │ L_s diagonal: min=$(round(minimum(d_L), sigdigits=4))  " *
               "max=$(round(maximum(d_L), sigdigits=4))  " *
               "mean=$(round(mean(d_L), sigdigits=4))")
    println(io, "    │ ‖L_sL_s'b−A_sb‖/‖A_sb‖: NeuralIF=$(round(rel_res, sigdigits=4))  " *
               "(Jacobi on orig ≈$(round(rel_jac, sigdigits=4)))")
end

"""
    print_neuralif_grad_probe(A, graph, params; n_rhs, rng, io=stdout)

Two independent Hutchinson batches on the **same** `(A, graph)` give gradients
`g₁`, `g₂` (flattened). Prints:

- `‖g₁‖₂`, `‖g₂‖₂` — gradient magnitudes (scale of the update signal).
- `cos(g₁, g₂)` — directional agreement (1 = same direction; near 0 / negative = noisy / unstable).
- `‖g₁+g₂‖₂ / ‖g₁−g₂‖₂` — stability ratio: large when two probes agree (small difference vs sum norm);
  small when Hutchinson noise dominates the difference between batches.

This is **not** the epoch `mean loss` (which averages over different matrices); it isolates
stochastic gradient noise on one instance.
"""
function print_neuralif_grad_probe(A::SparseMatrixCSC, graph::NeuralIFGraph, params;
                                   n_rhs::Int, rng::AbstractRNG, io::IO=stdout)
    N = size(A, 1)
    rhs1 = randn(rng, Float32, N, n_rhs)
    rhs2 = randn(rng, Float32, N, n_rhs)
    u = _neuralif_grad_flat(A, graph, params, rhs1)
    v = _neuralif_grad_flat(A, graph, params, rhs2)
    length(u) == length(v) || error("gradient structure mismatch in grad probe")
    nu = sqrt(sum(abs2, u))
    nv = sqrt(sum(abs2, v))
    cos_uv = dot(u, v) / (nu * nv + 1e-30)
    w = u .- v
    s = u .+ v
    snr = sqrt(sum(abs2, s)) / (sqrt(sum(abs2, w)) + 1e-30)
    println(io, "    │ grad probe (same A, two Hutchinson batches): " *
               "‖g₁‖₂=$(round(nu, sigdigits=4))  ‖g₂‖₂=$(round(nv, sigdigits=4))  " *
               "cos=$(round(cos_uv, sigdigits=4))  ‖g₁+g₂‖/‖g₁−g₂‖=$(round(snr, sigdigits=4))")
    return nothing
end

# ── Training on a problem class ───────────────────────────────────────────────

"""
    train_neuralif!(params, problem_class, cfg; kwargs...) -> params

Train the NeuralIF preconditioner on matrices sampled from `problem_class`.
Fresh matrices are drawn each epoch for implicit data augmentation.

# Keyword arguments
- `n_epochs`             : training epochs (default 100)
- `n_samples_per_epoch`  : matrices sampled per epoch (default 4)
- `lr`                   : Adam learning rate (default 3f-3)
- `n_rhs`                : Hutchinson probes per matrix per step (default 4)
- `verbose`              : print every N epochs; 0 = silent (default 20)
- `diagnostics`          : `false` (default) or `true`: at each verbose epoch, print operator
                          probe on the **last** matrix of the epoch plus a **gradient** probe
                          (‖g‖, cosine, ‖g₁+g₂‖/‖g₁−g₂‖) on that same matrix.
- `rng`                  : random number generator
"""
function train_neuralif!(params,
                         problem_class::AbstractProblemClass,
                         ::NeuralIFConfig;
                         n_epochs::Int             = 100,
                         n_samples_per_epoch::Int  = 4,
                         lr::Float32               = 1f-3,
                         n_rhs::Int                = 6,
                         verbose::Int              = 20,
                         diagnostics::Bool         = false,
                         rng::AbstractRNG          = Random.default_rng())
    opt       = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, params)

    for epoch in 1:n_epochs
        matrices   = [sample_matrix(problem_class, rng) for _ in 1:n_samples_per_epoch]
        epoch_loss = 0.0
        last_A     = nothing
        last_g     = nothing

        for A in matrices
            last_A = A
            graph  = build_neuralif_graph(A)
            last_g = graph
            N      = size(A, 1)
            rhs    = randn(rng, Float32, N, n_rhs)

            loss, grads = Zygote.withgradient(params) do ps
                _neuralif_step_loss(A, graph, ps, rhs)
            end

            opt_state, params = Optimisers.update!(opt_state, params, grads[1])
            epoch_loss += loss
        end

        if verbose > 0 && epoch % verbose == 0
            avg = epoch_loss / n_samples_per_epoch
            println("  Epoch $(lpad(epoch, 4)) | mean loss = $(round(avg, sigdigits=5))")
            if diagnostics && last_A !== nothing
                N     = size(last_A, 1)
                rhs_p = randn(rng, Float64, N, n_rhs)
                println("    │ held-out probe: n=$N  nnz=$(nnz(last_A))")
                print_neuralif_probe(last_A, last_g, params, rhs_p)
                print_neuralif_grad_probe(last_A, last_g, params; n_rhs=n_rhs, rng=rng)
            end
        end
    end

    return params
end

# ── Fine-tune on a specific matrix ────────────────────────────────────────────

"""
    fine_tune_neuralif!(params, A, cfg; kwargs...) -> params

Fine-tune a pre-trained NeuralIF on the target matrix A.

# Keyword arguments
- `n_steps`   : gradient steps (default 50)
- `lr`        : Adam learning rate (default 5f-4)
- `n_rhs`     : Hutchinson probes per step (default 8)
- `verbose`      : print every N steps; 0 = silent
- `diagnostics`  : `true` / `false` — same single probe block as `train_neuralif!`
"""
function fine_tune_neuralif!(params,
                              A::SparseMatrixCSC,
                              ::NeuralIFConfig;
                              n_steps::Int    = 50,
                              lr::Float32     = 5f-4,
                              n_rhs::Int      = 8,
                              verbose::Int    = 0,
                              diagnostics::Bool = false,
                              rng::AbstractRNG = Random.default_rng())
    opt       = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, params)
    graph     = build_neuralif_graph(A)
    N         = size(A, 1)

    for step in 1:n_steps
        rhs = randn(rng, Float32, N, n_rhs)
        loss, grads = Zygote.withgradient(params) do ps
            _neuralif_step_loss(A, graph, ps, rhs)
        end
        opt_state, params = Optimisers.update!(opt_state, params, grads[1])

        if verbose > 0 && step % verbose == 0
            println("  fine_tune step $(lpad(step, 4)) | loss = $(round(loss, sigdigits=5))")
            if diagnostics
                rhs_p = randn(rng, Float64, N, n_rhs)
                println("    │ probe (target A): n=$N  nnz=$(nnz(A))")
                print_neuralif_probe(A, graph, params, rhs_p)
                print_neuralif_grad_probe(A, graph, params; n_rhs=n_rhs, rng=rng)
            end
        end
    end

    return params
end
