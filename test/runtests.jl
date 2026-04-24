using NeuralPreconditioners
using Test
using LinearAlgebra
using SparseArrays
using Random
using Krylov
using Statistics

rng = Random.MersenneTwister(0)

@testset "NeuralPreconditioners.jl" begin

    # ── Problem generators ────────────────────────────────────────────────────
    @testset "Problems" begin
        A = poisson_2d(8)
        N = 64
        @test size(A) == (N, N)
        @test issymmetric(A)
        @test isposdef(Matrix(A))

        Acd = convection_diffusion_2d(8, 0.1, 1.0, 0.0)
        @test size(Acd) == (N, N)
        @test !issymmetric(Acd)

        rhs = generate_rhs(N, 5, rng)
        @test size(rhs) == (N, 5)
    end

    # ── Problem-class API ─────────────────────────────────────────────────────
    @testset "Problem classes" begin
        # PoissonClass
        cls_p = PoissonClass(grid_range=6:2:10)
        A = sample_matrix(cls_p, rng)
        @test A isa SparseMatrixCSC
        @test issymmetric(A)
        @test isposdef(Matrix(A))

        mats = generate_training_matrices(cls_p, 4, rng)
        @test length(mats) == 4
        @test all(m -> size(m, 1) == size(m, 2), mats)

        # HeterogeneousPoissonClass
        cls_h = HeterogeneousPoissonClass(grid_range=6:2:10, contrast_range=(2.0, 6.0))
        A_h = sample_matrix(cls_h, rng)
        # D*A0*D is mathematically SPD but D*A0*D[i,j] vs [j,i] differ in last bit
        @test norm(A_h - A_h', Inf) / norm(A_h, Inf) < 1e-12  # near-symmetric
        @test isposdef(Symmetric(Matrix(A_h)))                  # SPD after symmetrisation
        # Heterogeneous: diagonal entries should vary (not all identical)
        d = diag(A_h)
        @test maximum(d) / minimum(d) > 1.5

        # ConvectionDiffusionClass
        cls_cd = ConvectionDiffusionClass(grid_range=6:6, epsilon_range=(0.05, 0.1))
        A_cd = sample_matrix(cls_cd, rng)
        @test size(A_cd, 1) == 36
        @test !issymmetric(A_cd)
    end

    # ── Graph utilities ───────────────────────────────────────────────────────
    @testset "Graph construction" begin
        A = poisson_2d(8)
        g = build_graph(A)
        @test g.n == 64
        @test size(g.node_features, 1) == 4
        @test size(g.node_features, 2) == g.n
        @test size(g.A_hat) == (g.n, g.n)
        @test all(x -> 0 ≤ x ≤ 1.01, g.A_hat.nzval)
        @test length(g.diag_inv) == g.n
        @test all(g.diag_inv .> 0)
    end

    # ── GNN model ─────────────────────────────────────────────────────────────
    @testset "GNN preconditioner" begin
        A   = poisson_2d(8)
        g   = build_graph(A)
        cfg = GNNConfig(node_dim=4, hidden_dim=16, n_layers=2)
        ps  = init_gnn_params(rng, cfg)

        c = gnn_predict(g, ps)
        @test length(c) == g.n
        @test all(c .> 0)

        b = randn(rng, g.n)
        Mb = gnn_apply(g, b, ps)
        @test length(Mb) == g.n
    end

    # ── Transformer model ─────────────────────────────────────────────────────
    @testset "Block-diagonal transformer" begin
        A   = poisson_2d(8)
        cfg = TransformerConfig(block_size=8, hidden_dim=16, n_heads=2, n_layers=1)
        ps  = init_transformer_params(rng, cfg)

        M_blocks = transformer_predict(A, ps, cfg)
        K = cld(size(A, 1), cfg.block_size)
        @test size(M_blocks) == (K, cfg.block_size, cfg.block_size)

        b      = randn(rng, size(A, 1))
        result = transformer_apply(A, b, ps, cfg)
        @test length(result) == length(b)
    end

    # ── Loss functions ────────────────────────────────────────────────────────
    @testset "Loss functions" begin
        A   = poisson_2d(8)
        g   = build_graph(A)
        cfg = GNNConfig(node_dim=4, hidden_dim=16, n_layers=2)
        ps  = init_gnn_params(rng, cfg)
        rhs = randn(rng, 64, 6)

        c = gnn_predict(g, ps)
        d = Float64.(g.diag_inv) .* Float64.(c)

        # residual_loss: non-negative
        L_res = residual_loss(A, rhs, d)
        @test L_res ≥ 0

        # SAI cosine loss: L = 1 - cos ∈ [0, 2]; 0 = perfect, 2 = anti-aligned
        L_sai = sai_cosine_loss(A, rhs, d)
        @test 0f0 ≤ L_sai ≤ 2f0

        # Perfect Jacobi (d = diag_inv) should achieve near-zero residual loss
        # for this simple Poisson (just checks it doesn't error)
        L_jac_rel = jacobi_relative_loss(A, rhs, d, g.diag_inv)
        @test L_jac_rel ≥ 0
    end

    # ── Krylov.jl solver integration ──────────────────────────────────────────
    @testset "Krylov.jl integration" begin
        A = poisson_2d(8)
        b = randn(rng, size(A, 1))
        b ./= norm(b)

        # Unpreconditioned CG via Krylov.jl
        x_cg, stats_cg = Krylov.cg(A, b; atol=0.0, rtol=1e-8)
        @test stats_cg.solved
        @test norm(A * x_cg - b) / norm(b) < 1e-7

        # Jacobi-preconditioned via NeuralPreconditionerWrapper
        M_jac    = jacobi_preconditioner(A)
        wrapper  = NeuralPreconditionerWrapper(M_jac)
        x_jac, stats_jac = Krylov.cg(A, b; M=wrapper, atol=0.0, rtol=1e-8)
        @test stats_jac.solved
        @test stats_jac.niter ≤ stats_cg.niter   # Jacobi should not be worse

        # GNN preconditioner via NeuralPreconditionerWrapper (smoke test)
        cfg     = GNNConfig(node_dim=4, hidden_dim=16, n_layers=2)
        ps      = init_gnn_params(rng, cfg)
        g       = build_graph(A)
        M_gnn   = gnn_preconditioner(g, ps)
        wrapper_gnn = NeuralPreconditionerWrapper(M_gnn)
        x_gnn, stats_gnn = Krylov.cg(A, b; M=wrapper_gnn, atol=0.0, rtol=1e-8,
                                       itmax=500)
        @test norm(A * x_gnn - b) / norm(b) < 1e-6

        # Neumann-corrected GNN
        M_neu = gnn_neumann_preconditioner(A, g, ps; alpha=0.5)
        wrapper_neu = NeuralPreconditionerWrapper(M_neu)
        x_neu, stats_neu = Krylov.cg(A, b; M=wrapper_neu, atol=0.0, rtol=1e-8,
                                       itmax=500)
        @test norm(A * x_neu - b) / norm(b) < 1e-6
    end

    # ── Benchmark metrics ─────────────────────────────────────────────────────
    @testset "Benchmark (BenchmarkEntry fields)" begin
        A   = poisson_2d(8)
        rhs = randn(rng, 64, 4)
        M_jac = jacobi_preconditioner(A)
        entries = benchmark_preconditioners(A, rhs,
                      [("Jacobi", M_jac)]; tol=1e-8)
        e = entries[1]
        @test e.iters > 0
        @test e.rel_res < 1e-6
        @test e.converged ≈ 1.0
        @test e.total_time_s ≥ 0
        @test e.inference_time_s ≥ 0
        @test e.solve_time_s ≥ 0
        @test e.time_per_iter_s ≥ 0
    end

    # ── Training ──────────────────────────────────────────────────────────────
    @testset "Training: matrix-list (SAI cosine loss)" begin
        A   = poisson_2d(6)
        cfg = GNNConfig(node_dim=4, hidden_dim=8, n_layers=1)
        ps  = init_gnn_params(rng, cfg)
        g   = build_graph(A)
        rhs = randn(rng, 36, 4)

        d0 = Float64.(g.diag_inv) .* Float64.(gnn_predict(g, ps))
        L0 = sai_cosine_loss(A, rhs, d0)

        ps2 = train_preconditioner!(deepcopy(ps), [A], cfg;
                                     n_epochs=5, lr=1f-3, n_rhs=4,
                                     loss_type=:sai_cosine, verbose=0)

        d1 = Float64.(g.diag_inv) .* Float64.(gnn_predict(g, ps2))
        L1 = sai_cosine_loss(A, rhs, d1)
        @test L1 ≤ L0 + 0.15 * L0   # loss (1 - cos) should not increase significantly
    end

    @testset "Training: problem-class dispatch" begin
        cls = HeterogeneousPoissonClass(grid_range=6:2:8, contrast_range=(2.0, 5.0))
        cfg = GNNConfig(node_dim=4, hidden_dim=8, n_layers=1)
        ps  = init_gnn_params(rng, cfg)

        # Verify that problem-class dispatch runs without error
        ps2 = train_preconditioner!(deepcopy(ps), cls, cfg;
                                     n_epochs=3, n_samples_per_epoch=2,
                                     lr=1f-3, n_rhs=4, verbose=0, rng=rng)
        @test ps2 isa NamedTuple
    end

end
