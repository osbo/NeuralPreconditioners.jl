"""
examples/compare_preconditioners.jl

Full comparative benchmark across preconditioners and grid sizes.

Preconditioners compared
────────────────────────
  • CG (unpreconditioned)
  • Jacobi (diagonal scaling)
  • GNN diagonal (trained offline on HeterogeneousPoisson problem class, SAI cosine loss)
  • GNN diagonal (fine-tuned to each target)
  • GNN + one-step Neumann correction (fine-tuned)
  • Block-Diagonal Transformer

All solves use Krylov.cg (SPD systems only).
Benchmark reports: iters, rel residual, convergence %, total time,
                   inference time, solve time, time-per-iter.

Run:
    julia --project=. examples/compare_preconditioners.jl
"""

using NeuralPreconditioners
using LinearAlgebra
using SparseArrays
using Random
using Krylov
using Statistics

rng = Random.MersenneTwister(123)

println("=" ^ 70)
println("  NeuralPreconditioners.jl — Full preconditioner comparison")
println("=" ^ 70)

# ── 1. Define training problem class ─────────────────────────────────────────
println("\n[1] Training problem class: HeterogeneousPoisson …")
problem_class = HeterogeneousPoissonClass(
    grid_range     = 6:2:14,
    contrast_range = (2.0, 8.0),
)

# ── 2. Train GNN offline on problem class (SAI cosine loss) ──────────────────
println("[2] Training GNN (SAI cosine loss, 80 epochs, re-sampling each epoch) …")
cfg_gnn = GNNConfig(node_dim=4, hidden_dim=32, n_layers=3)
ps_gnn  = init_gnn_params(rng, cfg_gnn)
ps_gnn  = train_preconditioner!(ps_gnn, problem_class, cfg_gnn;
                                 n_epochs            = 60,
                                 n_samples_per_epoch = 3,
                                 lr                  = 3f-3,
                                 n_rhs               = 8,
                                 loss_type           = :sai_cosine,
                                 anchor_weight       = 2f-2,
                                 verbose             = 20,
                                 rng                 = rng)
ps_gnn  = train_preconditioner!(ps_gnn, problem_class, cfg_gnn;
                                 n_epochs            = 20,
                                 n_samples_per_epoch = 3,
                                 lr                  = 5f-4,
                                 n_rhs               = 12,
                                 loss_type           = :sai_cosine,
                                 anchor_weight       = 1f-2,
                                 verbose             = 0,
                                 rng                 = rng)
println("  GNN offline training complete.")

# ── 3. Train Block-Diagonal Transformer on sampled matrices ───────────────────
println("[3] Training block-diagonal transformer (40 epochs) …")
cfg_tr    = TransformerConfig(block_size=8, hidden_dim=32, n_heads=4, n_layers=2)
ps_tr     = init_transformer_params(rng, cfg_tr)
train_mats = generate_training_matrices(problem_class, 12, rng)
ps_tr      = train_transformer!(ps_tr, train_mats, cfg_tr;
                                n_epochs=40, lr=1f-3, n_rhs=4, verbose=0, rng=rng)
println("  Transformer training complete.\n")

# ── 4. Sweep over held-out heterogeneous test systems ────────────────────────
test_grid_sizes = [8, 12, 16, 20]
n_rhs_test      = 12
solve_tol       = 1e-10

for n_test in test_grid_sizes
    A = sample_matrix(
        HeterogeneousPoissonClass(grid_range=n_test:n_test, contrast_range=(4.0, 8.0)),
        rng)
    N = size(A, 1)
    println("─" ^ 70)
    println("Grid: $(n_test)×$(n_test)  (N = $N DOFs, nnz = $(nnz(A)))")
    println("─" ^ 70)

    rhs = generate_rhs(N, n_rhs_test, rng)
    g   = build_graph(A)

    # Fine-tune GNN and select Neumann α
    ps_ft = fine_tune!(deepcopy(ps_gnn), A, cfg_gnn;
                        n_steps=60, lr=5f-4, n_rhs=12,
                        loss_type=:sai_cosine, rng=rng)

    d_ft  = Float64.(g.diag_inv) .* Float64.(gnn_predict(g, ps_ft))
    α_best = 0.0
    best_iters = Inf
    for α in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        M = gnn_neumann_preconditioner(A, g, ps_ft; alpha=α)
        it = mean(pcg(A, rhs[:, j], M; tol=solve_tol, maxiter=2000).iters
                  for j in 1:min(n_rhs_test, 4))
        if it < best_iters; best_iters = it; α_best = α; end
    end

    preconditioners = [
        ("CG (unpreconditioned)",      identity),
        ("Jacobi",                     jacobi_preconditioner(A)),
        ("GNN offline (SAI)",          gnn_preconditioner(g, ps_gnn)),
        ("GNN fine-tuned (SAI)",       gnn_preconditioner(g, ps_ft)),
        ("GNN + Neumann (α=$(α_best))", gnn_neumann_preconditioner(A, g, ps_ft; alpha=α_best)),
        ("Block-Diag Transformer",     transformer_preconditioner(A, ps_tr, cfg_tr)),
    ]

    entries = benchmark_preconditioners(A, rhs, preconditioners;
                                         tol=solve_tol, maxiter=2000)
    print_benchmark_results(entries)

    κ_A, κ_MA = condition_number_ratio(A, d_ft)
    println("  κ(A) = $(round(κ_A, sigdigits=3)),  " *
            "κ(D^½AD^½) = $(round(κ_MA, sigdigits=3)),  " *
            "reduction = $(round(κ_MA/κ_A, digits=3))×\n")
end

println("Comparison complete.")
