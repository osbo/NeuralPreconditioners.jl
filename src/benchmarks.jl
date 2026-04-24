# ─────────────────────────────────────────────────────────────────────────────
# Benchmarking utilities
# ─────────────────────────────────────────────────────────────────────────────

"""
    BenchmarkEntry

Per-preconditioner benchmark statistics, averaged over all RHS samples.

Fields
──────
- `name`             : preconditioner label
- `iters`            : mean CG iterations
- `rel_res`          : mean final relative residual ‖Ax-b‖/‖b‖
- `converged`        : fraction of solves that converged (0–1)
- `total_time_s`     : mean wall-clock time for the full solve
- `inference_time_s` : mean time to apply M once (calibrated, excludes CG overhead)
- `solve_time_s`     : mean time attributed to CG arithmetic (total - iters×inference)
- `time_per_iter_s`  : mean total time per CG iteration
"""
struct BenchmarkEntry
    name             :: String
    iters            :: Float64
    rel_res          :: Float64
    converged        :: Float64
    total_time_s     :: Float64
    inference_time_s :: Float64
    solve_time_s     :: Float64
    time_per_iter_s  :: Float64
end

# ── Inference-time calibration ────────────────────────────────────────────────

function _calibrate_inference(M_apply::Function, b_probe::AbstractVector;
                               n_warmup::Int=5, n_timed::Int=20)
    for _ in 1:n_warmup
        M_apply(b_probe)
    end
    t = @elapsed for _ in 1:n_timed
        M_apply(b_probe)
    end
    return t / n_timed
end

# ── Main benchmark ────────────────────────────────────────────────────────────

"""
    benchmark_preconditioners(A, rhs_matrix, preconditioners;
                               tol, maxiter) -> Vector{BenchmarkEntry}

Benchmark each preconditioner on every column of `rhs_matrix` using Krylov.cg.

Requires A to be SPD (symmetric positive definite).  For non-symmetric systems
use the custom `pcg` wrapper or Krylov.gmres.

`preconditioners` is a `Vector` of `(name::String, M_apply::Function)` pairs where
`M_apply(v)` returns an approximation to A⁻¹v.

Timing breakdown
────────────────
- **inference_time**: calibrated by timing `M_apply` in isolation (excludes CG).
- **solve_time**: `total_time - iters × inference_time` (CG arithmetic, mat-vecs).
- **time_per_iter**: `total_time / iters`.
"""
function benchmark_preconditioners(A::SparseMatrixCSC,
                                    rhs_matrix::AbstractMatrix,
                                    preconditioners::AbstractVector;
                                    tol::Float64=1e-8,
                                    maxiter::Int=1000)
    m         = size(rhs_matrix, 2)
    b_probe   = rhs_matrix[:, 1]   # representative vector for inference timing
    entries   = BenchmarkEntry[]

    for (name, M_apply) in preconditioners
        # ── Calibrate single M application ──
        inf_time = _calibrate_inference(M_apply, b_probe)

        iters_v  = Float64[]
        res_v    = Float64[]
        time_v   = Float64[]
        conv_v   = Bool[]

        wrapper = NeuralPreconditionerWrapper(M_apply)

        for j in 1:m
            b = rhs_matrix[:, j]

            t_start = time()
            x, stats = Krylov.cg(A, b;
                                  M       = wrapper,
                                  atol    = 0.0,
                                  rtol    = tol,
                                  itmax   = maxiter)
            t_total = time() - t_start

            final_res = norm(A * x - b) / (norm(b) + 1e-15)
            push!(iters_v, stats.niter)
            push!(res_v,   final_res)
            push!(time_v,  t_total)
            push!(conv_v,  stats.solved)
        end

        mean_iters = mean(iters_v)
        mean_total = mean(time_v)
        mean_inf   = inf_time
        mean_solve = mean_total - mean_iters * mean_inf
        mean_piter = mean_iters > 0 ? mean_total / mean_iters : 0.0

        push!(entries, BenchmarkEntry(
            name,
            mean_iters,
            mean(res_v),
            mean(Float64.(conv_v)),
            mean_total,
            mean_inf,
            max(mean_solve, 0.0),   # clamp; calibration noise can make this tiny-negative
            mean_piter,
        ))
    end

    return entries
end

# ── Pretty-printing ───────────────────────────────────────────────────────────

"""
    print_benchmark_results(entries)

Pretty-print a benchmark results table with full timing breakdown.
"""
function print_benchmark_results(entries::AbstractVector{BenchmarkEntry})
    println()
    sep = "─"^100
    println(sep)
    @printf("%-32s %8s %14s %8s %10s %12s %10s %11s\n",
            "Preconditioner", "Iters", "Rel.Residual", "Conv(%)",
            "Total(s)", "Infer(s)", "Solve(s)", "s/iter")
    println(sep)
    for e in entries
        @printf("%-32s %8.1f %14.2e %8.1f %10.4f %12.2e %10.4f %11.2e\n",
                e.name,
                e.iters,
                e.rel_res,
                100 * e.converged,
                e.total_time_s,
                e.inference_time_s,
                e.solve_time_s,
                e.time_per_iter_s)
    end
    println(sep)
    println()
end

# ── Condition number analysis ─────────────────────────────────────────────────

"""
    condition_number_ratio(A, d; exact_threshold) -> (κ_A, κ_MA)

Compute κ(A) and κ(D^{1/2} A D^{1/2}) where D = diag(d).

The preconditioned system relevant for PCG convergence is D^{1/2} A D^{1/2}
(SPD ↔ exact eigenvalue spectrum).  For n ≤ `exact_threshold` (default 1000),
uses `eigvals` (exact); larger matrices fall back to a power-method estimate.
"""
function condition_number_ratio(A::SparseMatrixCSC, d::AbstractVector;
                                 exact_threshold::Int=1000)
    n   = size(A, 1)
    d64 = Float64.(d)

    function kappa(eigs)
        eigs_pos = filter(x -> x > 0, eigs)
        isempty(eigs_pos) && return Inf
        maximum(eigs_pos) / minimum(eigs_pos)
    end

    if n ≤ exact_threshold
        A_dense  = Symmetric(Matrix{Float64}(A))
        D_sqrt   = Diagonal(sqrt.(d64))
        MA_dense = Symmetric(Matrix{Float64}(D_sqrt * A * D_sqrt))
        κ_A  = kappa(eigvals(A_dense))
        κ_MA = kappa(eigvals(MA_dense))
    else
        rng_pw = Random.MersenneTwister(42)
        k      = 80

        function power_eig_max(op)
            v = randn(rng_pw, n); v ./= norm(v)
            λ = 0.0
            for _ in 1:k
                w = op(v); λ = dot(v, w); v = w ./ norm(w)
            end
            return abs(λ)
        end

        λ_max_A  = power_eig_max(v -> A * v)
        λ_min_A  = λ_max_A - power_eig_max(v -> λ_max_A .* v - A * v)
        κ_A      = λ_max_A / max(λ_min_A, 1e-14)

        MA_op    = v -> d64 .* (A * v)
        λ_max_MA = power_eig_max(MA_op)
        λ_min_MA = λ_max_MA - power_eig_max(v -> λ_max_MA .* v - MA_op(v))
        κ_MA     = λ_max_MA / max(λ_min_MA, 1e-14)
    end

    return κ_A, κ_MA
end
