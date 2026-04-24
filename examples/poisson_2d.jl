"""
examples/poisson_2d.jl

End-to-end demonstration of NeuralPreconditioners.jl on the 2D Poisson equation.

Workflow
────────
1. Define a HeterogeneousPoissonClass problem family (parameter ranges).
2. Train a GNN diagonal preconditioner offline with SAI cosine loss.
3. Fine-tune on the held-out target system.
4. Benchmark all methods directly via Krylov.cg:
   CG, Jacobi, Neumann-1..4, and neural preconditioners.

Run from the package root:
    julia --project=. examples/poisson_2d.jl
"""

using NeuralPreconditioners
using LinearAlgebra
using SparseArrays
using Random
using Krylov
using Statistics

# ── Reproducibility ───────────────────────────────────────────────────────────
rng = Random.MersenneTwister(42)

println("=" ^ 68)
println("  NeuralPreconditioners.jl — 2D Poisson demo (SAI cosine loss)")
println("=" ^ 68)

# ── 1. Define training problem class ─────────────────────────────────────────
println("\n[1] Problem class: HeterogeneousPoisson, grids 6–16, contrast 2–8 …")
problem_class = HeterogeneousPoissonClass(
    grid_range     = 6:2:16,
    contrast_range = (2.0, 8.0),
)

# ── 2. Configure and initialise the GNN ──────────────────────────────────────
println("\n[2] Initialising GNN preconditioner …")
cfg_gnn = GNNConfig(
    node_dim   = 4,
    hidden_dim = 64,
    n_layers   = 3,
)
ps_gnn = init_gnn_params(rng, cfg_gnn)
println("    embed($(cfg_gnn.node_dim)→$(cfg_gnn.hidden_dim)) + " *
        "$(cfg_gnn.n_layers) GCN layers + output head")

# ── 3. Offline training via problem class (SAI cosine loss) ───────────────────
println("\n[3] Offline training on problem class (200 epochs, SAI cosine loss) …")
# Phase 1: coarse descent — re-samples 4 fresh matrices each epoch
ps_gnn = train_preconditioner!(ps_gnn, problem_class, cfg_gnn;
                                n_epochs            = 120,
                                n_samples_per_epoch = 4,
                                lr                  = 3f-3,
                                n_rhs               = 8,
                                loss_type           = :sai_cosine,
                                anchor_weight       = 2f-2,
                                verbose             = 20,
                                rng                 = rng)
# Phase 2: fine LR
ps_gnn = train_preconditioner!(ps_gnn, problem_class, cfg_gnn;
                                n_epochs            = 80,
                                n_samples_per_epoch = 4,
                                lr                  = 1f-3,
                                n_rhs               = 12,
                                loss_type           = :sai_cosine,
                                anchor_weight       = 2f-2,
                                verbose             = 20,
                                rng                 = rng)

# ── 4. Target system (held-out) ───────────────────────────────────────────────
println("\n[4] Building target system (16×16 heterogeneous grid, N=256) …")
n_test  = 16
A_test  = sample_matrix(
    HeterogeneousPoissonClass(grid_range=16:16, contrast_range=(6.0, 8.0)), rng)
N_test  = size(A_test, 1)
println("    Matrix size: $N_test × $N_test, nnz = $(nnz(A_test))")

# ── 5. Fine-tune on target ────────────────────────────────────────────────────
println("\n[5] Fine-tuning on target (120 gradient steps, SAI cosine loss) …")
ps_ft = fine_tune!(deepcopy(ps_gnn), A_test, cfg_gnn;
                   n_steps      = 120,
                   lr           = 1f-3,
                   n_rhs        = 16,
                   loss_type    = :sai_cosine,
                   anchor_weight = 1f-2,
                   rng          = rng)

# ── 6. Build preconditioners ──────────────────────────────────────────────────
graph_test = build_graph(A_test)

M_jac = jacobi_preconditioner(A_test)
d_jac = 1.0 ./ diag(A_test)

d_gnn = Float64.(graph_test.diag_inv) .* Float64.(gnn_predict(graph_test, ps_gnn))
d_ft  = Float64.(graph_test.diag_inv) .* Float64.(gnn_predict(graph_test, ps_ft))

M_gnn = v -> d_gnn .* v
M_ft  = v -> d_ft  .* v

# Cross-validate Neumann alpha on a small probe set
function select_neumann_alpha(A, builder; alphas=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                              n_probe=6, tol=1e-10, maxiter=1000)
    rhs_probe = generate_rhs(size(A, 1), n_probe, rng)
    best_α, best_iters = first(alphas), Inf
    for α in alphas
        M = builder(α)
        iters = mean(pcg(A, rhs_probe[:, j], M; tol=tol, maxiter=maxiter).iters
                     for j in 1:n_probe)
        if iters < best_iters; best_α, best_iters = α, iters; end
    end
    return best_α
end

α_best = select_neumann_alpha(A_test,
    α -> gnn_neumann_preconditioner(A_test, graph_test, ps_ft; alpha=α))
println("    Selected Neumann α: $(round(α_best, digits=2))")
M_ft_neumann = gnn_neumann_preconditioner(A_test, graph_test, ps_ft; alpha=α_best)

# k-step Neumann correction built on a fixed diagonal D = diag(d)
function neumann_k_preconditioner(A::SparseMatrixCSC, d::AbstractVector;
                                  k::Int, alpha::Float64=1.0)
    return v -> begin
        y = d .* v
        for _ in 1:k
            y = y .+ alpha .* (d .* (v .- A * y))
        end
        y
    end
end

# ── 7. Benchmark ──────────────────────────────────────────────────────────────
println("\n[6] Benchmarking (20 RHS, tol=1e-10, Krylov.cg) …")
rhs_bench = generate_rhs(N_test, 20, rng)

preconditioners = [
    ("CG (unpreconditioned)",  identity),
    ("Jacobi",                 M_jac),
    ("Neumann-1 (Jacobi)",     neumann_k_preconditioner(A_test, d_jac; k=1, alpha=1.0)),
    ("Neumann-2 (Jacobi)",     neumann_k_preconditioner(A_test, d_jac; k=2, alpha=1.0)),
    ("Neumann-3 (Jacobi)",     neumann_k_preconditioner(A_test, d_jac; k=3, alpha=1.0)),
    ("Neumann-4 (Jacobi)",     neumann_k_preconditioner(A_test, d_jac; k=4, alpha=1.0)),
    ("GNN offline (SAI)",      M_gnn),
    ("GNN fine-tuned (SAI)",   M_ft),
    ("GNN fine-tuned + Neumann", M_ft_neumann),
]

entries = benchmark_preconditioners(A_test, rhs_bench, preconditioners;
                                     tol=1e-10, maxiter=2000)
print_benchmark_results(entries)

println("Done.")
