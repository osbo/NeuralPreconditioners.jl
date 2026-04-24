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
using LinearAlgebra
using SparseArrays
using Random
using Krylov
using Statistics

rng = Random.MersenneTwister(42)

println("=" ^ 68)
println("  NeuralPreconditioners.jl — 2D Poisson demo (NeuralIF)")
println("=" ^ 68)

# ── 1. Problem class ──────────────────────────────────────────────────────────
println("\n[1] Problem class: HeterogeneousPoisson, grids 6–16, contrast 2–8 …")
problem_class = HeterogeneousPoissonClass(
    grid_range     = 6:2:16,
    contrast_range = (2.0, 8.0),
)

# ── 2. Configure NeuralIF ─────────────────────────────────────────────────────
println("\n[2] Initialising NeuralIF preconditioner …")
cfg = NeuralIFConfig(
    n_node_features = 8,
    d_edge          = 32,
    n_layers        = 3,
    hidden_size     = 32,
    skip_connections = true,
)
ps = init_neuralif_params(rng, cfg)
println("    3-layer GraphNet, d_edge=32, hidden=32, Hutchinson Frobenius loss")

# ── 3. Offline training ───────────────────────────────────────────────────────
println("\n[3] Offline training on problem class …")
println("    Phase 1: 150 epochs × 4 matrices/epoch, lr=1e-3")
ps = train_neuralif!(ps, problem_class, cfg;
                     n_epochs            = 150,
                     n_samples_per_epoch = 4,
                     lr                  = 1f-3,
                     n_rhs               = 6,
                     verbose             = 25,
                     diagnostics         = true,
                     rng                 = rng)

println("    Phase 2: 50 epochs, lr=3e-4")
ps = train_neuralif!(ps, problem_class, cfg;
                     n_epochs            = 50,
                     n_samples_per_epoch = 4,
                     lr                  = 3f-4,
                     n_rhs               = 8,
                     verbose             = 25,
                     diagnostics         = true,
                     rng                 = rng)

# ── 4. Target system ──────────────────────────────────────────────────────────
println("\n[4] Building target system (16×16 grid, N=256) …")
A_test = sample_matrix(
    HeterogeneousPoissonClass(grid_range=16:16, contrast_range=(6.0, 8.0)), rng)
N_test = size(A_test, 1)
println("    Matrix size: $N_test × $N_test, nnz = $(nnz(A_test))")

# ── 5. Fine-tune on target ────────────────────────────────────────────────────
println("\n[5] Fine-tuning on target (60 steps) …")
ps_ft = fine_tune_neuralif!(deepcopy(ps), A_test, cfg;
                             n_steps     = 60,
                             lr          = 5f-4,
                             n_rhs       = 8,
                             verbose       = 20,
                             diagnostics = true,
                             rng           = rng)

# ── 6. Preconditioner diagnostics ────────────────────────────────────────────
println("\n[5b] Operator diagnostics on target A …")
graph_test  = build_neuralif_graph(A_test)
rhs_diag    = Float64.(randn(rng, Float32, N_test, 16))
println("    offline params:")
print_neuralif_probe(A_test, graph_test, ps,    rhs_diag)
println("    fine-tuned params:")
print_neuralif_probe(A_test, graph_test, ps_ft, rhs_diag)

# Condition numbers
d_jac   = 1.0 ./ diag(A_test)
κ_A, κ_jac = condition_number_ratio(A_test, d_jac)
println("    κ(A) = $(round(κ_A, sigdigits=6)),  κ(D^{-1}A) Jacobi: $(round(κ_jac, sigdigits=6))")

# ── 7. Build preconditioners ──────────────────────────────────────────────────
d_jac = 1.0 ./ diag(A_test)

function neumann_k(A, d; k, alpha=1.0)
    return v -> begin
        y = d .* v
        for _ in 1:k; y = y .+ alpha .* (d .* (v .- A * y)); end
        y
    end
end

M_neuralif = neuralif_preconditioner(A_test, ps_ft, cfg)

# ── 8. Benchmark ──────────────────────────────────────────────────────────────
println("\n[6] Benchmarking (20 RHS, tol=1e-10, Krylov.cg) …")
rhs_bench = generate_rhs(N_test, 20, rng)

preconditioners = [
    ("CG (unpreconditioned)",  identity),
    ("Jacobi",                 v -> d_jac .* v),
    ("Neumann-1 (Jacobi)",     neumann_k(A_test, d_jac; k=1)),
    ("Neumann-2 (Jacobi)",     neumann_k(A_test, d_jac; k=2)),
    ("Neumann-3 (Jacobi)",     neumann_k(A_test, d_jac; k=3)),
    ("Neumann-4 (Jacobi)",     neumann_k(A_test, d_jac; k=4)),
    ("NeuralIF",               M_neuralif),
]

entries = benchmark_preconditioners(A_test, rhs_bench, preconditioners;
                                     tol=1e-10, maxiter=2000,
                                     timing_infer_seconds=0.02,
                                     timing_solve_seconds=0.06,
                                     verbose=true)
print_benchmark_results(entries)

println("Done.")
