"""
NeuralPreconditioners.jl

A Julia framework for learning-based preconditioning of sparse linear systems.
Provides a problem-class API for training a Neural Incomplete Factorization (NeuralIF)
preconditioner that plugs directly into Krylov.jl solvers.
"""
module NeuralPreconditioners

using LinearAlgebra
using SparseArrays
using Random
using Serialization
using Statistics
using Printf
using BenchmarkTools
using NNlib: relu, softplus, scatter
using Zygote
using Optimisers
using ChainRulesCore
using Krylov
using CUDA

include("problems.jl")
include("graph_utils.jl")
include("models/neuralif.jl")
include("cuda_support.jl")
include("solvers.jl")
include("training.jl")
include("benchmarks.jl")
include("checkpoints.jl")

export
    # Problem generators
    poisson_2d,
    convection_diffusion_2d,
    generate_rhs,
    generate_training_data,
    # Problem-class API
    AbstractProblemClass,
    PoissonClass,
    HeterogeneousPoissonClass,
    ConvectionDiffusionClass,
    sample_matrix,
    generate_training_matrices,
    # Graph utilities
    SparseGraph,
    build_graph,
    NeuralIFGraph,
    build_neuralif_graph,
    # NeuralIF preconditioner
    NeuralIFConfig,
    init_neuralif_params,
    neuralif_forward,
    neuralif_build_L,
    # Training
    train_neuralif!,
    fine_tune_neuralif!,
    print_neuralif_probe,
    print_neuralif_grad_probe,
    # Solvers
    SolveResult,
    pcg,
    cg_unpreconditioned,
    jacobi_preconditioner,
    ssor_preconditioner,
    neuralif_preconditioner,
    NeuralPreconditionerWrapper,
    NeuralIFPreconditioner,
    # Checkpoints
    save_neuralif,
    load_neuralif,
    # Benchmarks
    BenchmarkEntry,
    benchmark_preconditioners,
    print_benchmark_results,
    condition_number_ratio,
    # CUDA utilities
    gpu_available,
    to_gpu,
    to_cpu

end
