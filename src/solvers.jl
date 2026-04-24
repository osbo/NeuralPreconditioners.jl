# ─────────────────────────────────────────────────────────────────────────────
# Iterative solvers
# ─────────────────────────────────────────────────────────────────────────────

"""
    SolveResult

Result of a linear solve.

Fields
──────
- `x`         : solution vector
- `iters`     : number of iterations taken
- `residual`  : final relative residual ‖Ax - b‖ / ‖b‖
- `converged` : whether convergence tolerance was reached
- `time_s`    : wall-clock solve time in seconds
"""
struct SolveResult
    x         :: Vector{Float64}
    iters     :: Int
    residual  :: Float64
    converged :: Bool
    time_s    :: Float64
end

# ── Preconditioned Conjugate Gradient ────────────────────────────────────────

"""
    pcg(A, b, M_apply; tol, maxiter) -> SolveResult

Preconditioned Conjugate Gradient (PCG) with a user-supplied preconditioner.

`M_apply(v)` should return an approximation to A⁻¹ v.

This implementation follows the standard PCG algorithm (Saad 2003, Alg. 6.18).
"""
function pcg(A::SparseMatrixCSC, b::AbstractVector,
             M_apply::Function;
             tol::Float64=1e-8,
             maxiter::Int=500)
    t0 = time()
    n  = length(b)
    x  = zeros(Float64, n)
    r  = copy(b)
    z  = M_apply(r)
    p  = copy(z)
    rz = dot(r, z)

    b_norm    = norm(b)
    converged = false
    iter      = 0

    # Use while loop to avoid Julia's for-loop variable scoping in hard local scope
    while iter < maxiter
        iter += 1
        Ap = A * p
        α  = rz / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap

        if norm(r) < tol * b_norm
            converged = true
            break
        end

        z      = M_apply(r)
        rz_new = dot(r, z)
        β      = rz_new / rz
        p     .= z .+ β .* p
        rz     = rz_new
    end

    return SolveResult(x, iter, norm(A * x - b) / b_norm, converged, time() - t0)
end

# ── Convenience wrappers ──────────────────────────────────────────────────────

"""
    cg_unpreconditioned(A, b; tol, maxiter) -> SolveResult

Standard (unpreconditioned) CG — baseline for comparison.
"""
function cg_unpreconditioned(A::SparseMatrixCSC, b::AbstractVector;
                              tol::Float64=1e-8, maxiter::Int=500)
    return pcg(A, b, identity; tol=tol, maxiter=maxiter)
end

"""
    jacobi_preconditioner(A) -> Function

Return a function that applies the Jacobi (diagonal) preconditioner: M v = v ./ diag(A).
"""
function jacobi_preconditioner(A::SparseMatrixCSC)
    d_inv = 1.0 ./ diag(A)
    return v -> d_inv .* v
end

"""
    ssor_preconditioner(A, ω) -> Function

Return a function that applies SSOR as a preconditioner for SPD systems.
ω = 1.0 is symmetric Gauss-Seidel; ω ∈ (1, 2) gives SOR-like acceleration.

SSOR inverse: M⁻¹ v = ω(2-ω) · (D + ωU)⁻¹ · D · (D + ωL)⁻¹ · v
where L = tril(A,-1) and U = triu(A,1).
"""
function ssor_preconditioner(A::SparseMatrixCSC, ω::Float64=1.0)
    d   = diag(A)
    D   = Diagonal(d)
    # Precompute the two triangular factors (sparse)
    DωL = D + ω .* tril(A, -1)   # D + ω L  (lower triangular)
    DωU = D + ω .* triu(A,  1)   # D + ω U  (upper triangular)
    scale = ω * (2.0 - ω)

    function apply(v)
        z1 = LowerTriangular(Matrix(DωL)) \ v         # forward solve
        z2 = d .* z1                                  # diagonal scaling
        z3 = UpperTriangular(Matrix(DωU)) \ z2        # backward solve
        return scale .* z3
    end

    return apply
end

"""
    gnn_preconditioner(graph, params) -> Function

Wrap a trained GNN into a preconditioner function suitable for `pcg`.
"""
function gnn_preconditioner(graph::SparseGraph, params)
    c = Float64.(gnn_predict(graph, params))
    d = Float64.(graph.diag_inv) .* c
    return v -> d .* v
end

"""
    gnn_neumann_preconditioner(A, graph, params; alpha=1.0) -> Function

One-step Neumann-corrected learned diagonal preconditioner:
    M(v) = y + α * D * (v - A*y),   y = D*v,   D = diag(d_gnn)

This adds one residual-correction sweep on top of the learned diagonal and can
capture some off-diagonal coupling while keeping the model lightweight.
"""
function gnn_neumann_preconditioner(A::SparseMatrixCSC,
                                    graph::SparseGraph, params;
                                    alpha::Float64=1.0)
    c = Float64.(gnn_predict(graph, params))
    d = Float64.(graph.diag_inv) .* c
    return v -> begin
        y = d .* v
        y .+ alpha .* (d .* (v .- A * y))
    end
end

"""
    transformer_preconditioner(A, params, cfg) -> Function

Wrap a trained transformer into a preconditioner function suitable for `pcg`.
The block-inverse matrices are precomputed once and reused across CG iterations.
"""
function transformer_preconditioner(A::SparseMatrixCSC,
                                     params, cfg::TransformerConfig)
    p = cfg.block_size
    n = size(A, 1)
    K = cld(n, p)

    blocks, _, _ = _extract_blocks(A, p)
    M_blocks = [Float64.(_encode_block(blocks[k, :, :], params)) for k in 1:K]

    function apply(v)
        result = zeros(Float64, n)
        for k in 1:K
            i_start = (k - 1) * p + 1
            i_end   = min(k * p, n)
            sz      = i_end - i_start + 1
            result[i_start:i_end] = M_blocks[k][1:sz, 1:sz] * v[i_start:i_end]
        end
        return result
    end

    return apply
end

# ── Krylov.jl-compatible preconditioner wrapper ───────────────────────────────

"""
    NeuralPreconditionerWrapper(apply_fn)

Thin wrapper that exposes a preconditioner function as a `LinearAlgebra.ldiv!`
object, making it a drop-in preconditioner for `Krylov.cg`, `Krylov.minres`,
`Krylov.gmres`, and all other Krylov.jl solvers.

Usage
─────
```julia
using Krylov
wrapper = NeuralPreconditionerWrapper(gnn_preconditioner(graph, ps))
x, stats = Krylov.cg(A, b; M = wrapper)
println(stats.niter, " iterations, residual = ", stats.residuals[end])
```
"""
struct NeuralPreconditionerWrapper
    apply_fn :: Function
end

# 5-arg mul!: y ← α·(M⁻¹ x) + β·y
# Krylov.jl calls this form by default (ldiv=false means mul! semantics)
function LinearAlgebra.mul!(y::AbstractVector,
                             P::NeuralPreconditionerWrapper,
                             x::AbstractVector,
                             α::Number, β::Number)
    Px = P.apply_fn(x)
    if iszero(β)
        y .= α .* Px
    else
        y .= α .* Px .+ β .* y
    end
    return y
end

# 3-arg mul!: y ← M⁻¹ x  (delegates to 5-arg)
function LinearAlgebra.mul!(y::AbstractVector,
                             P::NeuralPreconditionerWrapper,
                             x::AbstractVector)
    return LinearAlgebra.mul!(y, P, x, true, false)
end

# ldiv! forms for callers that use ldiv=true
function LinearAlgebra.ldiv!(y::AbstractVector,
                              P::NeuralPreconditionerWrapper,
                              x::AbstractVector)
    y .= P.apply_fn(x)
    return y
end

function LinearAlgebra.ldiv!(P::NeuralPreconditionerWrapper,
                              x::AbstractVector)
    x .= P.apply_fn(copy(x))
    return x
end
