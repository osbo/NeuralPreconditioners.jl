# NeuralPreconditioners.jl

**Learning-based preconditioning for sparse linear systems in Julia.**

Large simulations spend much of their time on iterative solves of sparse systems \(Ax=b\). Preconditioning reshapes the spectrum so Krylov methods need fewer iterations; neural approaches can learn rich approximations that classical fixed-sparsity factors miss. Published neural preconditioners, however, rarely share one training stack, one evaluation protocol, or one plug-in API—so comparisons are unfair and deployment into real codes is painful.

**NeuralPreconditioners.jl** is a small framework built around one contract: **a problem class** defines how training matrices are sampled; **shared machinery** handles Hutchinson-style self-supervised training (Adam + Zygote), GPU dispatch, diagnostics, checkpoints, and benchmarking; **NeuralIF** (Neural Incomplete Factorization) is the first architecture wired through end-to-end to **Krylov.jl** via standard `LinearAlgebra` preconditioner interfaces.

---

**Carl Osborne** · MIT EECS 6-4 · osbo@mit.edu  

**18.337 / 6.7320** — Final Project — Spring 2026

---

## What you get

| Piece | Role |
|--------|------|
| `AbstractProblemClass`, `sample_matrix` | Encode a *distribution* over matrices (`PoissonClass`, `HeterogeneousPoissonClass`, `ConvectionDiffusionClass`). |
| `build_neuralif_graph`, `NeuralIFGraph` | Turn `SparseMatrixCSC` into the lower-triangular graph NeuralIF consumes. |
| `train_neuralif!`, `fine_tune_neuralif!` | Shared training loop with configurable probes; optional CG iteration tracking on a validation matrix. |
| `NeuralIFPreconditioner` | One-shot forward → sparse \(L\); apply \((LL^\top)^{-1}\) with triangular solves; works as `M` in `Krylov.cg` or `Pl`-style APIs (e.g. LinearSolve.jl). |
| `save_neuralif` / `load_neuralif` | Offline train, reload elsewhere (`Serialization`; GPU weights saved from CPU copies). |
| `benchmark_preconditioners` | BenchmarkTools harness over RHS batches and named preconditioners. |

Helpers: `poisson_2d`, `convection_diffusion_2d`, `generate_rhs`, Jacobi/SSOR closures, `NeuralPreconditionerWrapper`, `gpu_available`, `to_gpu`, `to_cpu`.

---

## NeuralIF in one paragraph

Following [Häusner et al., arXiv:2305.16368](https://arxiv.org/abs/2305.16368), a graph network on the lower triangle of \(A\) predicts nonzero values of a sparse \(L\) with the **same pattern as** \(\mathrm{tril}(A)\). Training minimizes a **relative Hutchinson loss** in **Jacobi-scaled** space (\(\tilde{A} = D^{-1/2}AD^{-1/2}\), \(D=\mathrm{diag}(A)\)) so scales match across problems; diagonal outputs use \(\exp(z/2)\) for positivity near identity. At inference, the preconditioner undoes scaling outside two sparse triangular solves. Aggregation uses **`NNlib.scatter`** and inverse degrees—not dense scatter matrices—so memory stays \(\mathcal{O}(\mathrm{nnz})\).

---

## Requirements

- Julia **≥ 1.9** (see `Project.toml`).
- **CUDA.jl**: optional; CPU paths work without a GPU.

---

## Install

```julia
using Pkg
Pkg.develop(path="/path/to/NeuralPreconditioners.jl")
```

---

## Minimal solve

```julia
using NeuralPreconditioners, Krylov, Random, SparseArrays

rng = MersenneTwister(0)
A = poisson_2d(16)
b = randn(size(A, 1))

cfg = NeuralIFConfig(n_layers=3, d_edge=32, hidden_size=32)
ps  = init_neuralif_params(rng, cfg)

cls = HeterogeneousPoissonClass(grid_range=8:2:14, contrast_range=(2.0, 6.0))
ps  = train_neuralif!(ps, cls, cfg; n_epochs=20, n_samples_per_epoch=2,
                      n_rhs=6, lr=1f-3, verbose=5, rng=rng)

g = build_neuralif_graph(A)
M = NeuralIFPreconditioner(A, ps, cfg; prebuilt_graph=g)

x, stats = Krylov.cg(A, b; M=M, atol=0.0, rtol=1e-8)
```

Fine-tune on a target matrix: `fine_tune_neuralif!(deepcopy(ps), A, cfg; n_steps=60, val_A=A, …)`. Persist: `save_neuralif("ckpt.jls", ps, cfg)` then `ps, cfg = load_neuralif("ckpt.jls")`.

---

## Diagnostics

- **`print_neuralif_probe`:** Hutchinson loss, \(L\) diagonal stats, scaled relative residual vs. Jacobi.
- **`print_neuralif_grad_probe`:** Gradient agreement between two Hutchinson batches (cosine, \(\|g_1+g_2\|/\|g_1-g_2\|\)) to gauge noise vs. signal.

When `val_A` is provided during training, the code can track mean **CG iterations** and keep parameters that minimize that downstream metric.

---

## Examples & tests

```bash
julia --project=. examples/poisson_2d.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

`examples/poisson_2d.jl` walks through class setup, phased training, fine-tuning, checkpoint I/O, and benchmarking.

**Note:** Additional model sources live under `src/models/` (e.g. GNN diagonal, transformer sketches). They are **not** loaded by `using NeuralPreconditioners`; the supported public surface today is the NeuralIF pipeline listed in `src/NeuralPreconditioners.jl`.

---

## Limitations & next steps

- NeuralIF applies via **sparse triangular solves**, which are inherently sequential per level on GPU—same class of bottleneck as classical incomplete factorizations; architectures dominated by dense block ops may win raw GPU throughput for large \(N\).
- Hutchinson gradients are **noisy**; probe count and learning rate matter (`print_neuralif_grad_probe` helps).
- CG validation is the right early-stopping signal but costs more than loss-only steps.

Roadmap from the project report: port a hierarchical block transformer (\(\mathcal{H}\)-style partition); broaden problem classes (3D Poisson, anisotropic diffusion, elasticity); publish reproducible benchmarks and pretrained checkpoints.

---

## References

1. D. Häusner, O. Eberhard, J. Lässig. *Neural incomplete factorization.* arXiv:2305.16368, 2023.  
2. J. Chen. *Graph neural preconditioners for iterative solutions of sparse linear systems.* ICLR 2024.  
3. V. Trifonov et al. Learning preconditioners via graph networks (see report bibliography).  
4. Y. Li, W. Matusik et al. ICML 2023 preconditioner learning.  
5. [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) · [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) · M. Innes, *Don’t unroll adjoint*, arXiv:1810.07951.  
6. W. Hackbusch, *Hierarchical Matrices*, Springer, 2015.
