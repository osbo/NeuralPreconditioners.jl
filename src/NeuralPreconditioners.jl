"""
NeuralPreconditioners.jl

A Julia framework for learning-based preconditioning of sparse linear systems.
Provides a problem-class API for training GNN and transformer preconditioners
that plug directly into Krylov.jl solvers.
"""
module NeuralPreconditioners

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using Printf
using NNlib: relu, softplus, softmax
using Zygote
using Optimisers
using ChainRulesCore
using Krylov

include("problems.jl")
include("graph_utils.jl")
include("models/gnn.jl")
include("models/transformer.jl")
include("training.jl")
include("solvers.jl")
include("benchmarks.jl")

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
    # GNN preconditioner
    GNNConfig,
    init_gnn_params,
    gnn_predict,
    gnn_apply,
    gnn_preconditioner,
    gnn_neumann_preconditioner,
    # Block-diagonal transformer preconditioner
    TransformerConfig,
    init_transformer_params,
    transformer_predict,
    transformer_apply,
    transformer_preconditioner,
    # Loss functions
    sai_cosine_loss,
    residual_loss,
    jacobi_relative_loss,
    frobenius_loss,
    # Training
    train_preconditioner!,
    train_transformer!,
    fine_tune!,
    # Solvers
    SolveResult,
    pcg,
    cg_unpreconditioned,
    jacobi_preconditioner,
    ssor_preconditioner,
    NeuralPreconditionerWrapper,
    # Benchmarks
    BenchmarkEntry,
    benchmark_preconditioners,
    print_benchmark_results,
    condition_number_ratio

end
