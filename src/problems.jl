# ─────────────────────────────────────────────────────────────────────────────
# PDE problem generators and problem-class API
# ─────────────────────────────────────────────────────────────────────────────

"""
    poisson_2d(n) -> SparseMatrixCSC{Float64}

Assemble the 2D Poisson matrix -Δu = f on an n×n interior grid via the
5-point finite-difference stencil with homogeneous Dirichlet BCs.

Returns an N×N SPD matrix, N = n², with condition number O(h⁻²).
"""
function poisson_2d(n::Int)
    h = 1.0 / (n + 1)
    N = n^2
    rows, cols, vals = Int[], Int[], Float64[]

    for i in 1:n, j in 1:n
        k = (i - 1) * n + j
        push!(rows, k); push!(cols, k); push!(vals, 4.0 / h^2)

        if j < n
            push!(rows, k,   k+1)
            push!(cols, k+1, k  )
            push!(vals, -1.0/h^2, -1.0/h^2)
        end

        if i < n
            push!(rows, k,   k+n)
            push!(cols, k+n, k  )
            push!(vals, -1.0/h^2, -1.0/h^2)
        end
    end

    return sparse(rows, cols, vals, N, N)
end

"""
    convection_diffusion_2d(n, ε, cx, cy) -> SparseMatrixCSC{Float64}

Assemble -ε Δu + cₓ ∂ₓu + c_y ∂_y u = f on an n×n interior grid.
"""
function convection_diffusion_2d(n::Int, ε::Float64=0.01,
                                  cx::Float64=1.0, cy::Float64=0.0)
    h = 1.0 / (n + 1)
    N = n^2
    rows, cols, vals = Int[], Int[], Float64[]

    for i in 1:n, j in 1:n
        k = (i - 1) * n + j
        push!(rows, k); push!(cols, k); push!(vals, 4.0 * ε / h^2)

        if j < n
            kr = k + 1
            push!(rows, k,  kr )
            push!(cols, kr, k  )
            push!(vals, -ε/h^2 + cx/(2h),
                        -ε/h^2 - cx/(2h))
        end

        if i < n
            ku = k + n
            push!(rows, k,  ku )
            push!(cols, ku, k  )
            push!(vals, -ε/h^2 + cy/(2h),
                        -ε/h^2 - cy/(2h))
        end
    end

    return sparse(rows, cols, vals, N, N)
end

"""
    generate_rhs(N, n_samples[, rng]) -> Matrix{Float64}

Draw `n_samples` random right-hand side vectors of length N.
"""
function generate_rhs(N::Int, n_samples::Int,
                      rng::AbstractRNG=Random.default_rng())
    return randn(rng, Float64, N, n_samples)
end

"""
    generate_training_data(grid_sizes, n_per_size[, rng]) -> Vector{SparseMatrixCSC}

Generate a family of Poisson matrices at the specified grid sizes.
"""
function generate_training_data(grid_sizes::AbstractVector{Int},
                                 n_per_size::Int=1,
                                 rng::AbstractRNG=Random.default_rng())
    matrices = SparseMatrixCSC{Float64, Int}[]
    for n in grid_sizes, _ in 1:n_per_size
        A = poisson_2d(n)
        push!(matrices, (1.0 + 0.1 * randn(rng)) * A)
    end
    return matrices
end

# ── Internal generator ────────────────────────────────────────────────────────

"""
    _heterogeneous_poisson_2d(n, rng; contrast) -> SparseMatrixCSC{Float64}

Construct an SPD matrix A = D·A₀·D where D = diag(s) and s > 0 is a smooth
random scaling field.  The contrast ‖s‖_∞ / ‖s‖_min is clamped to `contrast`.
"""
function _heterogeneous_poisson_2d(n::Int, rng::AbstractRNG; contrast::Float64=8.0)
    A0 = poisson_2d(n)
    N  = n^2

    phase_x = 2π * rand(rng)
    phase_y = 2π * rand(rng)
    amp1    = 0.6 * randn(rng)
    amp2    = 0.6 * randn(rng)

    s = Vector{Float64}(undef, N)
    idx = 1
    for i in 1:n, j in 1:n
        x = i / (n + 1)
        y = j / (n + 1)
        log_s = amp1 * sin(2π * x + phase_x) + amp2 * cos(2π * y + phase_y) +
                0.15 * randn(rng)
        s[idx] = exp(log_s)
        idx += 1
    end

    smin = 1.0 / contrast
    smax = contrast
    s .= clamp.(s, smin, smax)

    D = Diagonal(s)
    return SparseMatrixCSC(D * A0 * D)
end

# ── Problem-class API ─────────────────────────────────────────────────────────

"""
    AbstractProblemClass

Abstract type for parameterised problem families.  Subtypes encode a
distribution over sparse linear systems from which training instances are drawn.

Implement `sample_matrix(class, rng) -> SparseMatrixCSC` for each subtype.

## Example

```julia
cls = HeterogeneousPoissonClass(grid_range=6:2:20, contrast_range=(2.0, 10.0))
A   = sample_matrix(cls, rng)               # one instance
mats = generate_training_matrices(cls, 20, rng)  # 20 instances
ps   = train_preconditioner!(ps, cls, cfg; n_epochs=100, n_samples_per_epoch=4)
```
"""
abstract type AbstractProblemClass end

"""
    sample_matrix(class, rng) -> SparseMatrixCSC

Draw one training matrix from the problem class distribution.
"""
function sample_matrix end

"""
    generate_training_matrices(class, n_samples[, rng]) -> Vector{SparseMatrixCSC}

Sample `n_samples` matrices from `class`.
"""
function generate_training_matrices(class::AbstractProblemClass, n_samples::Int,
                                     rng::AbstractRNG=Random.default_rng())
    return [sample_matrix(class, rng) for _ in 1:n_samples]
end

# ── Concrete problem classes ──────────────────────────────────────────────────

"""
    PoissonClass(; grid_range)

Uniform 2D Poisson matrices on grids n×n, n ∈ grid_range.
"""
Base.@kwdef struct PoissonClass <: AbstractProblemClass
    grid_range :: AbstractRange{Int} = 6:2:16
end

function sample_matrix(cls::PoissonClass, rng::AbstractRNG)
    n = rand(rng, cls.grid_range)
    return poisson_2d(n)
end

"""
    HeterogeneousPoissonClass(; grid_range, contrast_range)

2D Poisson matrices with spatially varying coefficients: A = D·A₀·D,
D = diag(s), s drawn from a smooth random field.

- `grid_range`     : range of grid sizes n (N = n² unknowns)
- `contrast_range` : (lo, hi) for the max/min scaling ratio; sampled uniformly
"""
Base.@kwdef struct HeterogeneousPoissonClass <: AbstractProblemClass
    grid_range     :: AbstractRange{Int}      = 6:2:16
    contrast_range :: Tuple{Float64, Float64} = (2.0, 10.0)
end

function sample_matrix(cls::HeterogeneousPoissonClass, rng::AbstractRNG)
    n        = rand(rng, cls.grid_range)
    lo, hi   = cls.contrast_range
    contrast = lo + rand(rng) * (hi - lo)
    return _heterogeneous_poisson_2d(n, rng; contrast=contrast)
end

"""
    ConvectionDiffusionClass(; grid_range, epsilon_range, cx_range, cy_range)

2D convection-diffusion matrices -ε Δu + c·∇u = f with random parameters.
Note: these matrices are generally non-symmetric; use GMRES-compatible preconditioners.
"""
Base.@kwdef struct ConvectionDiffusionClass <: AbstractProblemClass
    grid_range    :: AbstractRange{Int}        = 6:2:16
    epsilon_range :: Tuple{Float64, Float64}   = (0.01, 0.1)
    cx_range      :: Tuple{Float64, Float64}   = (-1.0, 1.0)
    cy_range      :: Tuple{Float64, Float64}   = (-1.0, 1.0)
end

function sample_matrix(cls::ConvectionDiffusionClass, rng::AbstractRNG)
    n  = rand(rng, cls.grid_range)
    ε  = cls.epsilon_range[1] + rand(rng) * (cls.epsilon_range[2] - cls.epsilon_range[1])
    cx = cls.cx_range[1]  + rand(rng) * (cls.cx_range[2]  - cls.cx_range[1])
    cy = cls.cy_range[1]  + rand(rng) * (cls.cy_range[2]  - cls.cy_range[1])
    return convection_diffusion_2d(n, ε, cx, cy)
end
