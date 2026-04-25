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
- `setup_time_s`     : optional **one-shot** setup / inference / factorisation work (neural forward, AMG hierarchy
                       build, …), from a caller-supplied `setup_thunk` on `(name, M, setup_thunk)` rows; `0` when
                       omitted (e.g. Jacobi has no meaningful setup).
- `solve_time_s`     : `mean(total) - mean(iters) × τ` where `τ` is the minimum sampled time for **one**
                       preconditioner apply `M(v)` on a probe vector (PCG bookkeeping, not printed separately).
- `time_per_iter_s`  : `mean(total) / mean(iters)` using the robust `total_time_s` above
"""
struct BenchmarkEntry
    name             :: String
    iters            :: Float64
    rel_res          :: Float64
    converged        :: Float64
    total_time_s   :: Float64
    setup_time_s   :: Float64
    solve_time_s   :: Float64
    time_per_iter_s  :: Float64
end

# ── Krylov tolerances (must match real(eltype(A)) / eltype(b) on GPU) ─────────

function _cg_atol_rtol(A, b, tol::Real)
    T = promote_type(real(eltype(A)), real(eltype(b)))
    return zero(T), T(tol)
end

# ── BenchmarkTools calibration ───────────────────────────────────────────────

function _parse_preconditioner_row(spec::Tuple)
    n = length(spec)
    n == 2 && return spec[1], spec[2], nothing
    n == 3 && return spec[1], spec[2], spec[3]
    throw(ArgumentError(
        "each preconditioner must be (name, M) or (name, M, setup_thunk); got a tuple of length $n",
    ))
end

function _benchmark_minimum_seconds(f, seconds::Float64)::Float64
    trial = @benchmark $(f)() evals=1 seconds=seconds
    return minimum(trial.times) * 1e-9
end

function _minimum_time_m_apply(M_apply, b_probe::AbstractVector;
                               timing_mapply_seconds::Float64, use_cuda::Bool)
    thunk = if use_cuda
        () -> CUDA.@sync(M_apply(b_probe))
    else
        () -> M_apply(b_probe)
    end
    return _benchmark_minimum_seconds(thunk, timing_mapply_seconds)
end

function _minimum_time_setup(setup_thunk;
                            timing_setup_seconds::Float64, use_cuda::Bool)
    thunk = if use_cuda
        () -> CUDA.@sync(setup_thunk())
    else
        () -> setup_thunk()
    end
    return _benchmark_minimum_seconds(thunk, timing_setup_seconds)
end

function _cg_solve_minimum_time(A, b::AbstractVector, wrapper;
                               tol::Real, maxiter::Int,
                               timing_solve_seconds::Float64, use_cuda::Bool)::Float64
    atol, rtol = _cg_atol_rtol(A, b, tol)
    if use_cuda
        thunk = () -> CUDA.@sync Krylov.cg(A, b; M=wrapper, atol=atol, rtol=rtol, itmax=maxiter)
    else
        thunk = () -> Krylov.cg(A, b; M=wrapper, atol=atol, rtol=rtol, itmax=maxiter)
    end
    trial = @benchmark $(thunk)() evals=1 seconds=timing_solve_seconds
    return minimum(trial.times) * 1e-9
end

# ── Main benchmark ────────────────────────────────────────────────────────────

"""
    benchmark_preconditioners(A, rhs_matrix, preconditioners;
                              tol, maxiter,
                              timing_setup_seconds, timing_mapply_seconds,
                              timing_solve_seconds,
                              verbose=false) -> Vector{BenchmarkEntry}

Benchmark each preconditioner on every column of `rhs_matrix` using `Krylov.cg`.

Requires A to be SPD (symmetric positive definite).  For non-symmetric systems
use the custom `pcg` wrapper or Krylov.gmres.

`preconditioners` is a vector of either:

- `(name::String, M_apply)` — `M_apply(v)` is the Krylov preconditioner; **setup** column is `0`.
- `(name::String, M_apply, setup_thunk)` — same `M_apply`, and `setup_thunk()` (no arguments) is timed
  once for **setup / inference / factorisation** (e.g. neural forward, AMG hierarchy build). Jacobi-style
  methods typically omit the third element.

`timing_mapply_seconds` budgets BenchmarkTools sampling for **one** apply `M_apply(b_probe)`; that
minimum time `τ` is used only to derive `solve_time_s ≈ mean(total) - mean(iters)×τ` (PCG bookkeeping).

## Timing (`BenchmarkTools.jl`)

Uses `@benchmark` with `evals=1` (each sample is one full `cg` solve or one setup / one `M` apply) and reports
`minimum(trial.times)` in seconds. The `seconds` keywords budget wall time spent **sampling** each
benchmark, so BenchmarkTools can auto-tune `samples` within that budget.

**Wall time (why it can be much larger than the budgets):** for each of `P` preconditioners and each
of `m` RHS columns, the code (1) runs **one full** `Krylov.cg` for iteration counts and residuals,
then (2) runs `@benchmark` with `seconds=timing_solve_seconds` (each sample is a **complete** CG
with `evals=1`). So you pay roughly `P × m` full solves **before** the timed sampling loop, plus
`P × m` benchmark runs that each hold the clock open for about `timing_solve_seconds` (plus warmup).
On GPU, `CUDA.@sync` runs every timed sample. Different preconditioner closures are different Julia
types, so the first pass can pay extra **JIT / kernel specialization** cost. Lower `m`, lower both
`timing_*` values, or call `benchmark_preconditioners` once as a warmup if you need a fast demo.

Across multiple RHS columns, `total_time_s` is the **mean** of the per-column minima (summary over
the RHS batch).

- `verbose=true`: print each preconditioner name to `stderr` when its sampling starts (helps when
  the loop sits quietly for a long time).

Timing breakdown
────────────────
- **setup_time_s**: optional one-shot `setup_thunk` (third tuple element); otherwise `0`.
- **solve_time_s**: `mean(total) - mean(iters) × τ` with `τ` from one `M_apply` sample (not the setup column).
- **time_per_iter**: `total_time / iters`.
"""
function benchmark_preconditioners(A,
                                    rhs_matrix::AbstractMatrix,
                                    preconditioners::AbstractVector;
                                    tol::Real=1e-8,
                                    maxiter::Int=1000,
                                    timing_setup_seconds::Float64=0.1,
                                    timing_mapply_seconds::Float64=0.1,
                                    timing_solve_seconds::Float64=0.25,
                                    verbose::Bool=false,
                                    use_cuda::Bool=false)
    m         = size(rhs_matrix, 2)
    b_probe   = rhs_matrix[:, 1]   # representative vector for M-apply calibration
    entries   = BenchmarkEntry[]

    for spec in preconditioners
        name, M_apply, setup_thunk = _parse_preconditioner_row(spec)
        verbose && println(stderr, "    benchmarking: ", name, " …")

        setup_time = if setup_thunk === nothing
            0.0
        else
            _minimum_time_setup(setup_thunk;
                                timing_setup_seconds=timing_setup_seconds,
                                use_cuda=use_cuda)
        end

        tau_m = _minimum_time_m_apply(M_apply, b_probe;
                                     timing_mapply_seconds=timing_mapply_seconds,
                                     use_cuda=use_cuda)

        iters_v  = Float64[]
        res_v    = Float64[]
        time_v   = Float64[]
        conv_v   = Bool[]

        wrapper = NeuralPreconditionerWrapper(M_apply)

        for j in 1:m
            b = rhs_matrix[:, j]
            atol, rtol = _cg_atol_rtol(A, b, tol)

            x, stats = Krylov.cg(A, b; M=wrapper, atol=atol, rtol=rtol, itmax=maxiter)
            use_cuda && CUDA.synchronize()
            final_res = norm(A * x - b) / (norm(b) + 1e-15)

            t_min = _cg_solve_minimum_time(A, b, wrapper; tol=tol, maxiter=maxiter,
                                           timing_solve_seconds=timing_solve_seconds,
                                           use_cuda=use_cuda)

            push!(iters_v, stats.niter)
            push!(res_v,   final_res)
            push!(time_v,  t_min)
            push!(conv_v,  stats.solved)
        end

        mean_iters = mean(iters_v)
        mean_total = mean(time_v)
        mean_solve = mean_total - mean_iters * tau_m
        mean_piter = mean_iters > 0 ? mean_total / mean_iters : 0.0

        push!(entries, BenchmarkEntry(
            name,
            mean_iters,
            mean(res_v),
            mean(Float64.(conv_v)),
            mean_total,
            setup_time,
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
            "Total(s)", "Setup(s)", "Solve(s)", "s/iter")
    println(io, sep)
    for e in entries
        @printf(io, "%-32s %8.1f %14.2e %8.1f %10.4e %12.2e %10.4e %11.2e\n",
                e.name,
                e.iters,
                e.rel_res,
                100 * e.converged,
                e.total_time_s,
                e.setup_time_s,
                e.solve_time_s,
                e.time_per_iter_s)
    end
    println(io, sep)
    println(io, "  Total / Setup / Solve: `BenchmarkTools.minimum(trial.times)`; tune sampling with ",
            "`timing_setup_seconds`, `timing_mapply_seconds`, and `timing_solve_seconds` on `benchmark_preconditioners`.")
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
