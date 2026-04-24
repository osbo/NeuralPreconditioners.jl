# ─────────────────────────────────────────────────────────────────────────────
# Sparse-matrix → graph representation
# ─────────────────────────────────────────────────────────────────────────────

"""
    SparseGraph

Graph representation of a sparse matrix A used by GNN preconditioners.

Fields
──────
- `n`            : number of nodes (= size(A,1))
- `nnz_count`    : number of structural nonzeros
- `row_idx`      : row index of each nonzero (1-based)
- `col_idx`      : column index of each nonzero
- `nz_vals`      : original nonzero values (Float32)
- `A_hat`        : symmetrically normalised adjacency with self-loops (Float32 sparse)
                   Â = D̃^{-1/2} (|A|+I) D̃^{-1/2}
- `node_features`: 4×n Float32 matrix — [std_diag, std_log_rownorm, std_log_deg, sign_diag]
- `diag_inv`     : Jacobi baseline diagonal inverse, 1 ./ diag(A) (Float32)
"""
struct SparseGraph
    n            :: Int
    nnz_count    :: Int
    row_idx      :: Vector{Int}
    col_idx      :: Vector{Int}
    nz_vals      :: Vector{Float32}
    A_hat        :: SparseMatrixCSC{Float32, Int}
    node_features:: Matrix{Float32}   # d_node × n
    diag_inv     :: Vector{Float32}
end

"""
    build_graph(A) -> SparseGraph

Construct a `SparseGraph` from a sparse matrix A.

The normalised adjacency uses |A| + I so it works for indefinite and
non-symmetric operators:
    Ã = |A| + I
    D̃ᵢᵢ = Σⱼ Ãᵢⱼ
    Â = D̃^{-1/2} Ã D̃^{-1/2}
"""
function build_graph(A::SparseMatrixCSC{T}) where {T<:Real}
    n = size(A, 1)
    @assert size(A, 2) == n "A must be square"

    ri, ci, nz = findnz(A)

    # ── Node features (d_node = 4) ──────────────────────────────────────────
    diag_vals  = Vector{Float64}(diag(A))
    row_norms  = vec(sqrt.(sum(abs2.(A), dims=2)))
    col_counts = Float64.(diff(A.colptr))   # nnz per column ≈ degree

    sign_diag = Float64.(sign.(diag_vals))

    function standardise(v::Vector{Float64})
        μ, σ = mean(v), std(v)
        return Float32.((v .- μ) ./ (σ + 1e-6))
    end

    node_features = vcat(
        standardise(diag_vals)',
        standardise(log.(row_norms .+ 1e-12))',
        standardise(log.(col_counts .+ 1.0))',
        Float32.(sign_diag)'
    )  # 4 × n

    # ── Normalised adjacency Â (Float32 sparse) ──────────────────────────────
    abs_A   = abs.(A)
    A_tilde = abs_A + I                        # SparseMatrixCSC with self-loops
    d_tilde = vec(sum(A_tilde, dims=2))
    d_inv_sqrt = 1.0 ./ sqrt.(d_tilde .+ 1e-12)
    D_inv_sqrt = Diagonal(d_inv_sqrt)
    A_hat_f64  = D_inv_sqrt * A_tilde * D_inv_sqrt
    A_hat      = SparseMatrixCSC{Float32, Int}(A_hat_f64)

    # Jacobi baseline (safe inverse for tiny/zero diagonal entries)
    diag_inv = Float32.(1.0 ./ max.(abs.(diag_vals), 1e-12))

    return SparseGraph(n, length(nz), ri, ci, Float32.(nz), A_hat, node_features, diag_inv)
end
