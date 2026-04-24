# ─────────────────────────────────────────────────────────────────────────────
# Iterative solvers and preconditioner wrappers
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

SSOR preconditioner for SPD systems. ω=1 is symmetric Gauss-Seidel.
"""
function ssor_preconditioner(A::SparseMatrixCSC, ω::Float64=1.0)
    d   = diag(A)
    D   = Diagonal(d)
    DωL = D + ω .* tril(A, -1)
    DωU = D + ω .* triu(A,  1)
    scale = ω * (2.0 - ω)
    function apply(v)
        z1 = LowerTriangular(Matrix(DωL)) \ v
        z2 = d .* z1
        z3 = UpperTriangular(Matrix(DωU)) \ z2
        return scale .* z3
    end
    return apply
end

# ── NeuralIF preconditioner ───────────────────────────────────────────────────

"""
    neuralif_preconditioner(A, params, cfg) -> Function

Build a NeuralIF preconditioner for matrix A.

Runs the GNN once to predict L values, assembles the sparse Cholesky factor L,
then returns a closure that applies (LL')⁻¹ via two sparse triangular solves.

The closure captures L and L' as `LowerTriangular` / `UpperTriangular` wrappers
so each PCG iteration costs O(nnz(L)) ≈ O(nnz(tril(A))).
"""
function neuralif_preconditioner(A::SparseMatrixCSC,
                                 params,
                                 ::NeuralIFConfig)
    graph  = build_neuralif_graph(A)
    L_vals = Float64.(neuralif_forward(graph, params))
    L_sp   = neuralif_build_L(L_vals, graph)   # sparse lower triangular (in scaled space)

    L_lower = LowerTriangular(L_sp)
    L_upper = UpperTriangular(sparse(L_sp'))   # L_s' as upper triangular

    # graph.d_sqrt_inv = 1/sqrt(diag(A))
    # M^{-1} r = D^{-1/2} (L_s L_s')^{-1} D^{-1/2} r
    d = graph.d_sqrt_inv

    return v -> begin
        r_s = d .* Float64.(v)    # D^{-1/2} r
        z   = L_lower \ r_s        # L_s z = r_s
        y_s = L_upper \ z          # L_s' y_s = z
        d .* y_s                   # D^{-1/2} y_s
    end
end

# ── Krylov.jl-compatible preconditioner wrapper ───────────────────────────────

"""
    NeuralPreconditionerWrapper(apply_fn)

Thin wrapper that exposes a preconditioner function as a `LinearAlgebra.ldiv!`
object, making it a drop-in preconditioner for `Krylov.cg` and other solvers.
"""
struct NeuralPreconditionerWrapper
    apply_fn :: Function
end

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

function LinearAlgebra.mul!(y::AbstractVector,
                             P::NeuralPreconditionerWrapper,
                             x::AbstractVector)
    return LinearAlgebra.mul!(y, P, x, true, false)
end

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
