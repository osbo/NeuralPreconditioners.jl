# NeuralPreconditioners.jl

Learning-based preconditioning for sparse linear systems, built as a Julia framework around one contract: a problem class supplies a distribution over matrices, shared machinery handles training and dispatch, and a `Preconditioner` type plugs into `Krylov.jl` like any classical factorisation.

## Why

Krylov solvers on large sparse systems spend most of their iterations paying for poor conditioning. Neural preconditioners (Häusner et al. 2023; Li et al. 2023) can learn richer factors than fixed-sparsity ILU at training time, then run cheaply at inference. Results in the field are hard to compare because every paper ships its own training stack, its own evaluation harness, and its own way of plugging into a solver. This package is the shared backbone they all could use.

## What's in it

| Piece | Role |
|---|---|
| `AbstractProblemClass`, `sample_matrix` | A distribution over `SparseMatrixCSC`s. Built-in: `PoissonClass`, `HeterogeneousPoissonClass`, `ConvectionDiffusionClass`. |
| `build_neuralif_graph`, `NeuralIFGraph` | Lowers an SPD matrix into NeuralIF's lower-triangular graph representation. |
| `train_neuralif!`, `fine_tune_neuralif!` | Hutchinson-style self-supervised training loop, with optional CG-iteration probes on a held-out matrix. |
| `NeuralIFPreconditioner` | Forward pass → sparse $L$; applies $(LL^\top)^{-1}$ via triangular solves; satisfies `Krylov.cg`'s `M` and LinearSolve.jl's `Pl` interfaces. |
| `save_neuralif` / `load_neuralif` | Offline training, reload elsewhere. GPU weights round-trip via CPU copies. |

## How training works

1. **Sample** — at each step, `sample_matrix(class)` draws a fresh SPD matrix from the problem class. Discretisation grid, coefficient field, and right-hand-side are randomised inside the class.
2. **Predict** — the NeuralIF graph network produces a sparse lower-triangular $L$ matching the lower triangle of $A$.
3. **Hutchinson loss** — the objective is the Frobenius residual of $(LL^\top)^{-1}A - I$, estimated by Hutchinson's trick with a handful of Rademacher probes. No reference factorisation is required.
4. **Probe** — every N steps, a validation matrix is solved with `cg(A, b; M = NeuralIFPreconditioner(model, A))` and the iteration count is logged.

The same loop drives `fine_tune_neuralif!`, which freezes the global graph weights and adapts a small residual head to the test-time problem class.

## Stack

- **Julia** — Flux, Zygote, CUDA.jl, SparseArrays, Krylov.jl
- **Models** — NeuralIF graph network in `src/models/`
- **Examples** — `examples/` runs end-to-end training + CG benchmarks

## Reports

- `NeuralPreconditioners_Report.pdf` — full write-up
- `NeuralPreconditioners_Report.tex` — LaTeX source
- `NeuralPreconditioners_Presentation.pptx` — slides

18.337 / 6.7320 — Spring 2026.

## License

MIT.
