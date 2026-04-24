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
- `total_time_s`     : mean over RHS of **BenchmarkTools** minimum sampled time for one full `cg` solve
- `inference_time_s` : **BenchmarkTools** `minimum(trial.times)` for one `M_apply` on a probe vector
- `solve_time_s`     : `mean(total) - mean(iters) × infer` (same residual accounting as before)
- `time_per_iter_s`  : `mean(total) / mean(iters)` using the robust `total_time_s` above
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

# ── BenchmarkTools calibration ───────────────────────────────────────────────

function _calibrate_inference(M_apply::Function, b_probe::AbstractVector;
                              timing_infer_seconds::Float64)
    thunk = () -> M_apply(b_probe)
    trial = @benchmark $(thunk)() evals=1 seconds=timing_infer_seconds
    return minimum(trial.times) * 1e-9
end

function _cg_solve_minimum_time(A::SparseMatrixCSC, b::AbstractVector, wrapper;
                               tol::Float64, maxiter::Int, timing_solve_seconds::Float64)::Float64
    thunk = () -> Krylov.cg(A, b; M=wrapper, atol=0.0, rtol=tol, itmax=maxiter)
    trial = @benchmark $(thunk)() evals=1 seconds=timing_solve_seconds
    return minimum(trial.times) * 1e-9
end

# ── Main benchmark ────────────────────────────────────────────────────────────

"""
    benchmark_preconditioners(A, rhs_matrix, preconditioners;
                              tol, maxiter,
                              timing_infer_seconds,
                              timing_solve_seconds,
                              verbose=false) -> Vector{BenchmarkEntry}

Benchmark each preconditioner on every column of `rhs_matrix` using `Krylov.cg`.

Requires A to be SPD (symmetric positive definite).  For non-symmetric systems
use the custom `pcg` wrapper or Krylov.gmres.

`preconditioners` is a `Vector` of `(name::String, M_apply::Function)` pairs where
`M_apply(v)` returns an approximation to A⁻¹v.

## Timing (`BenchmarkTools.jl`)

Uses `@benchmark` with `evals=1` (each sample is one full apply or one full `cg` solve) and reports
`minimum(trial.times)` in seconds. The `seconds` keyword budgets wall time spent **sampling** per
benchmark (`timing_infer_seconds` for `M_apply`, `timing_solve_seconds` **per RHS column** for `cg`),
so BenchmarkTools can auto-tune `samples` within that budget.

**Wall time (rough upper bound):** for `P` preconditioners and `m` RHS columns, expect on the order of
`P × (timing_infer_seconds + m × timing_solve_seconds)` seconds before the table prints — often a
few tens of seconds with defaults. Lower the two `timing_*` budgets or use fewer RHS if this feels
too slow.

Across multiple RHS columns, `total_time_s` is the **mean** of the per-column minima (summary over
the RHS batch).

- `verbose=true`: print each preconditioner name to `stderr` when its sampling starts (helps when
  the loop sits quietly for a long time).

Timing breakdown
────────────────
- **inference_time**: minimum sampled apply time (see above).
- **solve_time**: `total_time - iters × inference_time` (CG arithmetic, mat-vecs; residual accounting).
- **time_per_iter**: `total_time / iters`.
"""
function benchmark_preconditioners(A::SparseMatrixCSC,
                                    rhs_matrix::AbstractMatrix,
                                    preconditioners::AbstractVector;
                                    tol::Float64=1e-8,
                                    maxiter::Int=1000,
                                    timing_infer_seconds::Float64=0.1,
                                    timing_solve_seconds::Float64=0.25,
                                    verbose::Bool=false)
    m         = size(rhs_matrix, 2)
    b_probe   = rhs_matrix[:, 1]   # representative vector for inference timing
    entries   = BenchmarkEntry[]

    for (name, M_apply) in preconditioners
        verbose && println(stderr, "    benchmarking: ", name, " …")
        inf_time = _calibrate_inference(M_apply, b_probe;
                                        timing_infer_seconds=timing_infer_seconds)

        iters_v  = Float64[]
        res_v    = Float64[]
        time_v   = Float64[]
        conv_v   = Bool[]

        wrapper = NeuralPreconditionerWrapper(M_apply)

        for j in 1:m
            b = rhs_matrix[:, j]

            x, stats = Krylov.cg(A, b; M=wrapper, atol=0.0, rtol=tol, itmax=maxiter)
            final_res = norm(A * x - b) / (norm(b) + 1e-15)

            t_min = _cg_solve_minimum_time(A, b, wrapper; tol=tol, maxiter=maxiter,
                                           timing_solve_seconds=timing_solve_seconds)

            push!(iters_v, stats.niter)
            push!(res_v,   final_res)
            push!(time_v,  t_min)
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
            max(mean_solve, 0.0),
            mean_piter,
        ))
    end

    return entries
end

# ── Pretty-printing ───────────────────────────────────────────────────────────

"""
    print_benchmark_results(entries; io)

Pretty-print a benchmark results table with full timing breakdown.
"""
function print_benchmark_results(entries::AbstractVector{BenchmarkEntry}; io::IO=stdout)
    println(io)
    sep = "─"^100
    println(io, sep)
    @printf(io, "%-32s %8s %14s %8s %10s %12s %10s %11s\n",
            "Preconditioner", "Iters", "Rel.Residual", "Conv(%)",
            "Total(s)", "Infer(s)", "Solve(s)", "s/iter")
    println(io, sep)
    for e in entries
        @printf(io, "%-32s %8.1f %14.2e %8.1f %10.4e %12.2e %10.4e %11.2e\n",
                e.name,
                e.iters,
                e.rel_res,
                100 * e.converged,
                e.total_time_s,
                e.inference_time_s,
                e.solve_time_s,
                e.time_per_iter_s)
    end
    println(io, sep)
    println(io, "  Total / Infer: `BenchmarkTools.minimum(trial.times)`; tune sampling with ",
            "`timing_infer_seconds` / `timing_solve_seconds` on `benchmark_preconditioners`.")
    println(io)
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
