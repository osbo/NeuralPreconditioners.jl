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
        cls_p = PoissonClass(grid_range=6:2:10)
        A = sample_matrix(cls_p, rng)
        @test A isa SparseMatrixCSC
        @test issymmetric(A)
        @test isposdef(Matrix(A))

        mats = generate_training_matrices(cls_p, 4, rng)
        @test length(mats) == 4
        @test all(m -> size(m, 1) == size(m, 2), mats)

        cls_h = HeterogeneousPoissonClass(grid_range=6:2:10, contrast_range=(2.0, 6.0))
        A_h = sample_matrix(cls_h, rng)
        @test norm(A_h - A_h', Inf) / norm(A_h, Inf) < 1e-12
        @test isposdef(Symmetric(Matrix(A_h)))
        d = diag(A_h)
        @test maximum(d) / minimum(d) > 1.5

        cls_cd = ConvectionDiffusionClass(grid_range=6:6, epsilon_range=(0.05, 0.1))
        A_cd = sample_matrix(cls_cd, rng)
        @test size(A_cd, 1) == 36
        @test !issymmetric(A_cd)
    end

    # ── Graph utilities (SparseGraph for diagnostics / future use) ────────────
    @testset "Graph construction" begin
        A = poisson_2d(8)
        g = build_graph(A)
        @test g.n == 64
        @test size(g.node_features, 1) == 4
        @test size(g.node_features, 2) == g.n
        @test size(g.A_hat) == (g.n, g.n)
        @test size(g.A_msg) == (g.n, g.n)
        @test all(x -> 0 ≤ x ≤ 1.01, g.A_hat.nzval)
        @test length(g.diag_inv) == g.n
        @test all(g.diag_inv .> 0)
    end

    # ── NeuralIF model ───────────────────────────────────────────────────────
    @testset "NeuralIF forward and apply" begin
        A   = poisson_2d(8)
        cfg = NeuralIFConfig(n_node_features=8, d_edge=16, n_layers=3,
                             hidden_size=16, skip_connections=true)
        ps  = init_neuralif_params(rng, cfg)
        g   = build_neuralif_graph(A)
        Lv  = neuralif_forward(g, ps)
        @test length(Lv) == nnz(tril(A))
        Lsp = neuralif_build_L(Lv, g)
        @test size(Lsp) == (size(A, 1), size(A, 1))
    end

    # ── Krylov.jl integration ─────────────────────────────────────────────────
    @testset "Krylov.jl integration" begin
        A = poisson_2d(8)
        b = randn(rng, size(A, 1))
        b ./= norm(b)

        x_cg, stats_cg = Krylov.cg(A, b; atol=0.0, rtol=1e-8)
        @test stats_cg.solved
        @test norm(A * x_cg - b) / norm(b) < 1e-7

        M_jac = jacobi_preconditioner(A)
        wrapper = NeuralPreconditionerWrapper(M_jac)
        x_jac, stats_jac = Krylov.cg(A, b; M=wrapper, atol=0.0, rtol=1e-8)
        @test stats_jac.solved
        @test stats_jac.niter ≤ stats_cg.niter

        cfg = NeuralIFConfig(d_edge=16, hidden_size=16, n_layers=3)
        ps  = init_neuralif_params(rng, cfg)
        M_if = neuralif_preconditioner(A, ps, cfg)
        wrapper_if = NeuralPreconditionerWrapper(M_if)
        x_if, stats_if = Krylov.cg(A, b; M=wrapper_if, atol=0.0, rtol=1e-8, itmax=500)
        @test norm(A * x_if - b) / norm(b) < 1e-5
    end

    # ── Benchmark metrics ─────────────────────────────────────────────────────
    @testset "Benchmark (BenchmarkEntry fields)" begin
        A   = poisson_2d(8)
        rhs = randn(rng, 64, 4)
        M_jac = jacobi_preconditioner(A)
        entries = benchmark_preconditioners(A, rhs,
                      [("Jacobi", M_jac)]; tol=1e-8,
                      timing_setup_seconds=0.02,
                      timing_mapply_seconds=0.02,
                      timing_solve_seconds=0.06)
        e = entries[1]
        @test e.iters > 0
        @test e.rel_res < 1e-6
        @test e.converged ≈ 1.0
        @test e.total_time_s ≥ 0
        @test e.setup_time_s == 0  # no optional setup_thunk on (name, M) rows
        @test e.solve_time_s ≥ 0
        @test e.time_per_iter_s ≥ 0
    end

    # ── Training (NeuralIF) ───────────────────────────────────────────────────
    @testset "NeuralIF training smoke" begin
        cls = HeterogeneousPoissonClass(grid_range=6:6, contrast_range=(2.0, 4.0))
        cfg = NeuralIFConfig(d_edge=16, hidden_size=16, n_layers=3)
        ps  = init_neuralif_params(rng, cfg)
        ps2 = train_neuralif!(deepcopy(ps), cls, cfg;
                              n_epochs=2, n_samples_per_epoch=1,
                              lr=1f-3, n_rhs=4, verbose=0, diagnostics=false, rng=rng)
        @test ps2 isa NamedTuple

        A = sample_matrix(cls, rng)
        ps3 = fine_tune_neuralif!(deepcopy(ps2), A, cfg;
                                 n_steps=2, lr=1f-3, n_rhs=4,
                                 verbose=0, diagnostics=false, rng=rng)
        @test ps3 isa NamedTuple
    end

    @testset "NeuralIF grad probe" begin
        A = poisson_2d(6)
        g = build_neuralif_graph(A)
        cfg = NeuralIFConfig(d_edge=16, hidden_size=16, n_layers=3)
        ps = init_neuralif_params(rng, cfg)
        io = IOBuffer()
        print_neuralif_grad_probe(A, g, ps; n_rhs=4, rng=rng, io=io)
        s = String(take!(io))
        @test occursin("cos=", s)
        @test occursin("‖g₁+g₂‖/‖g₁−g₂‖=", s)
    end

end
