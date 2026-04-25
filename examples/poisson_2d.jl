"""
examples/poisson_2d.jl

End-to-end demonstration of NeuralPreconditioners.jl on the 2D Poisson equation.

Implements the Neural Incomplete Factorization (NeuralIF) preconditioner from
Häusner et al. (arXiv:2305.16368). The GNN learns a sparse Cholesky factor L
with the same sparsity pattern as tril(A), trained via a Hutchinson Frobenius
loss  E_w[‖LL'w − Aw‖₂]. Inference applies (LL')⁻¹ via two triangular solves.

Workflow
────────
1. Define a HeterogeneousPoissonClass problem family.
2. Train NeuralIF offline on sampled matrices (Hutchinson loss).
3. Fine-tune on the held-out target system.
4. Benchmark: CG, Jacobi, Neumann-1..4, and NeuralIF.

Run from the package root:
    julia --project=. examples/poisson_2d.jl
"""

using NeuralPreconditioners
using CUDA
using LinearAlgebra
using SparseArrays
using Random
using Krylov
using Statistics

_tsync() = (CUDA.functional() && CUDA.synchronize(); nothing)
_fmt(t)  = t < 60 ? "$(round(t, digits=2))s" : "$(round(t/60, digits=2))min"

# Section [8] uses BenchmarkTools per (preconditioner, RHS). Cost scales roughly as
#   (#preconditioners) × n_rhs × (one full CG + ~timing_solve_seconds of sampling each).
# Increase for stabler statistics; decrease for a quicker demo (defaults below are demo-friendly).
const N_BENCH_RHS = 4
const BENCH_TIMING_SETUP_S = 0.015   # optional setup_thunk (e.g. neural forward) — BenchmarkTools budget
const BENCH_TIMING_MAPPly_S = 0.015 # one M(v) apply — used only for Solve(s) bookkeeping vs Total(s)
const BENCH_TIMING_SOLVE_S = 0.02

rng = Random.MersenneTwister(42)

println("=" ^ 68)
println("  NeuralPreconditioners.jl — 2D Poisson demo (NeuralIF)")
println("=" ^ 68)

use_cuda = gpu_available()
if use_cuda
    println("  GPU detected — training and fine-tuning will run on CUDA.")
else
    println("  No GPU detected — running on CPU.")
end

# ── 1. Problem class ──────────────────────────────────────────────────────────
println("\n[1] Problem class: HeterogeneousPoisson, grids 6–16, contrast 2–8 …")
t1 = @elapsed begin
    problem_class = HeterogeneousPoissonClass(
        grid_range     = 12:2:28,
        contrast_range = (2.0, 8.0),
    )
end
println("    → $(_fmt(t1))")

# ── 2. Configure NeuralIF ─────────────────────────────────────────────────────
println("\n[2] Initialising NeuralIF preconditioner …")
t2 = @elapsed begin
    cfg = NeuralIFConfig(
        n_node_features  = 8,
        d_edge           = 32,
        n_layers         = 3,
        hidden_size      = 32,
        skip_connections = true,
    )
    ps = init_neuralif_params(rng, cfg)
end
println("    3-layer GraphNet, d_edge=32, hidden=32, Hutchinson Frobenius loss")
println("    → $(_fmt(t2))")

# ── 3. Offline training ───────────────────────────────────────────────────────
println("\n[3] Offline training on problem class …")
println("    Phase 1: 150 epochs × 4 matrices/epoch, lr=1e-3")
t3a = @elapsed begin
    ps = train_neuralif!(ps, problem_class, cfg;
                         n_epochs            = 150,
                         n_samples_per_epoch = 4,
                         lr                  = 1f-3,
                         n_rhs               = 6,
                         verbose             = 25,
                         diagnostics         = true,
                         rng                 = rng,
                         use_cuda            = use_cuda)
    _tsync()
end
println("    Phase 1 → $(_fmt(t3a))")

println("    Phase 2: 50 epochs, lr=3e-4")
t3b = @elapsed begin
    ps = train_neuralif!(ps, problem_class, cfg;
                         n_epochs            = 50,
                         n_samples_per_epoch = 4,
                         lr                  = 3f-4,
                         n_rhs               = 8,
                         verbose             = 25,
                         diagnostics         = true,
                         rng                 = rng,
                         use_cuda            = use_cuda)
    _tsync()
end
println("    Phase 2 → $(_fmt(t3b))  |  training total → $(_fmt(t3a + t3b))")

# ── 4. Target system ──────────────────────────────────────────────────────────
println("\n[4] Building target system (32×32 grid, N=1024) …")
t4 = @elapsed begin
    A_test = sample_matrix(
        HeterogeneousPoissonClass(grid_range=32:32, contrast_range=(6.0, 8.0)), rng)
    N_test = size(A_test, 1)
    graph_test = build_neuralif_graph(A_test)   # built once; reused by fine-tune, diagnostics, preconditioner
end
println("    Matrix size: $N_test × $N_test, nnz = $(nnz(A_test))")
println("    → $(_fmt(t4))")

# ── 5. Fine-tune on target ────────────────────────────────────────────────────
println("\n[5] Fine-tuning on target (60 steps) …")
t5 = @elapsed begin
    ps_ft = fine_tune_neuralif!(deepcopy(ps), A_test, cfg;
                                prebuilt_graph = graph_test,
                                n_steps        = 60,
                                lr             = 5f-4,
                                n_rhs          = 8,
                                verbose        = 20,
                                diagnostics    = true,
                                val_A          = A_test,
                                rng            = rng,
                                use_cuda       = use_cuda)
    _tsync()
end
println("    Fine-tune → $(_fmt(t5))")

# ── 5b. Operator diagnostics ──────────────────────────────────────────────────
println("\n[5b] Operator diagnostics on target A …")
t5b = @elapsed begin
    rhs_diag = Float64.(randn(rng, Float32, N_test, 16))
    println("    offline params:")
    print_neuralif_probe(A_test, graph_test, ps,    rhs_diag)
    println("    fine-tuned params:")
    print_neuralif_probe(A_test, graph_test, ps_ft, rhs_diag)

    d_jac  = 1.0 ./ diag(A_test)
    κ_A, κ_jac = condition_number_ratio(A_test, d_jac)
    println("    κ(A) = $(round(κ_A, sigdigits=6)),  κ(D^{-1}A) Jacobi: $(round(κ_jac, sigdigits=6))")
end
println("    Diagnostics → $(_fmt(t5b))")

# ── 6. Checkpoint trained params ─────────────────────────────────────────────
println("\n[6] Saving / loading checkpoint …")
t6 = @elapsed begin
    ckpt_path = tempname() * ".jls"
    save_neuralif(ckpt_path, ps_ft, cfg)
    ps_loaded, cfg_loaded = load_neuralif(ckpt_path)
    ps_loaded = use_cuda ? to_gpu(ps_loaded) : ps_loaded
    rm(ckpt_path)
end
println("    Checkpoint round-trip → $(_fmt(t6))")

# ── 7. Build preconditioners ──────────────────────────────────────────────────
println("\n[7] Building preconditioners …")
t7 = @elapsed begin
    d_jac = 1.0 ./ diag(A_test)

    function neumann_k(A, d; k, alpha=1.0)
        return v -> begin
            y = d .* v
            for _ in 1:k; y = y .+ alpha .* (d .* (v .- A * y)); end
            y
        end
    end

    # NeuralIFPreconditioner: user-facing type, compatible with Krylov.jl (M=)
    # and LinearSolve.jl (Pl=). Reuses the already-built graph_test.
    M_neuralif = NeuralIFPreconditioner(A_test, ps_ft, cfg_loaded;
                                         prebuilt_graph=graph_test)
    # One-shot GNN forward (same op as inside preconditioner build), timed in the Setup(s) column.
    neuralif_setup_thunk = if use_cuda
        let gd = to_gpu(graph_test), ps = ps_ft
            () -> (CUDA.@sync(neuralif_forward(gd, ps)); nothing)
        end
    else
        let g = graph_test, ps = ps_ft
            () -> (neuralif_forward(g, ps); nothing)
        end
    end
    _tsync()
end
println("    → $(_fmt(t7))")

# ── 8. Benchmark ──────────────────────────────────────────────────────────────
println("\n[8] Benchmarking ($N_BENCH_RHS RHS, tol=1e-10, Krylov.cg) …")
t8 = @elapsed begin
    rhs_bench = generate_rhs(N_test, N_BENCH_RHS, rng)

    if use_cuda
        A_bench       = CUDA.cu(A_test)
        rhs_bench_dev = CUDA.cu(Float64.(rhs_bench))
        d_jac_gpu     = CUDA.cu(d_jac)

        preconditioners = [
            ("CG (unpreconditioned)",  identity),
            ("Jacobi",                 v -> d_jac_gpu .* v),
            ("Neumann-1 (Jacobi)",     neumann_k(A_bench, d_jac_gpu; k=1)),
            ("Neumann-2 (Jacobi)",     neumann_k(A_bench, d_jac_gpu; k=2)),
            ("Neumann-3 (Jacobi)",     neumann_k(A_bench, d_jac_gpu; k=3)),
            ("Neumann-4 (Jacobi)",     neumann_k(A_bench, d_jac_gpu; k=4)),
            ("NeuralIF",               M_neuralif, neuralif_setup_thunk),
        ]

        entries = benchmark_preconditioners(A_bench, rhs_bench_dev, preconditioners;
                                             tol=1e-10, maxiter=2000,
                                             timing_setup_seconds=BENCH_TIMING_SETUP_S,
                                             timing_mapply_seconds=BENCH_TIMING_MAPPly_S,
                                             timing_solve_seconds=BENCH_TIMING_SOLVE_S,
                                             verbose=true,
                                             use_cuda=true)
    else
        preconditioners = [
            ("CG (unpreconditioned)",  identity),
            ("Jacobi",                 v -> d_jac .* v),
            ("Neumann-1 (Jacobi)",     neumann_k(A_test, d_jac; k=1)),
            ("Neumann-2 (Jacobi)",     neumann_k(A_test, d_jac; k=2)),
            ("Neumann-3 (Jacobi)",     neumann_k(A_test, d_jac; k=3)),
            ("Neumann-4 (Jacobi)",     neumann_k(A_test, d_jac; k=4)),
            ("NeuralIF",               M_neuralif, neuralif_setup_thunk),
        ]

        entries = benchmark_preconditioners(A_test, rhs_bench, preconditioners;
                                             tol=1e-10, maxiter=2000,
                                             timing_setup_seconds=BENCH_TIMING_SETUP_S,
                                             timing_mapply_seconds=BENCH_TIMING_MAPPly_S,
                                             timing_solve_seconds=BENCH_TIMING_SOLVE_S,
                                             verbose=true)
    end
    _tsync()
end
print_benchmark_results(entries)
println("    Benchmark → $(_fmt(t8))")

# ── Summary ───────────────────────────────────────────────────────────────────
println()
println("=" ^ 68)
println("  Timing summary")
println("=" ^ 68)
println("  Problem class setup   $(_fmt(t1))")
println("  Model init            $(_fmt(t2))")
println("  Offline train Ph.1    $(_fmt(t3a))")
println("  Offline train Ph.2    $(_fmt(t3b))")
println("  Graph + target build  $(_fmt(t4))")
println("  Fine-tune             $(_fmt(t5))")
println("  Diagnostics           $(_fmt(t5b))")
println("  Checkpoint round-trip $(_fmt(t6))")
println("  Preconditioner build  $(_fmt(t7))")
println("  Benchmark             $(_fmt(t8))")
total = t1+t2+t3a+t3b+t4+t5+t5b+t6+t7+t8
println("  ─────────────────────────────")
println("  Total                 $(_fmt(total))")
println("=" ^ 68)
println("Done.")
