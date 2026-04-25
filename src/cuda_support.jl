# ─────────────────────────────────────────────────────────────────────────────
# CUDA support: device transfer utilities for NeuralIFGraph and params
#
# All hot-path operations (GNN forward, Hutchinson loss, scatter-add) are
# device-agnostic via AbstractArray dispatch.  Call to_gpu before training to
# move data to the GPU; to_cpu to retrieve results.
# ─────────────────────────────────────────────────────────────────────────────

"""
    gpu_available() -> Bool

Return true if a CUDA-capable GPU is detected and functional.
"""
gpu_available() = CUDA.functional()

# ── Recursive array conversion for NamedTuple param trees ────────────────────

"""
    to_gpu(x)

Recursively move all arrays in `x` (NamedTuple, Tuple, or AbstractArray) to
the GPU.  Scalars and `nothing` pass through unchanged.  If no GPU is
available this will raise a CUDA error; check `gpu_available()` first.
"""
to_gpu(a::AbstractArray)  = CUDA.cu(a)
to_gpu(::Nothing)         = nothing
to_gpu(x::Number)         = x
to_gpu(nt::NamedTuple)    = map(to_gpu, nt)
to_gpu(t::Tuple)          = map(to_gpu, t)

"""
    to_cpu(x)

Recursively move all arrays in `x` back to the CPU.
"""
to_cpu(a::Array)          = a
to_cpu(a::AbstractArray)  = Array(a)
to_cpu(::Nothing)         = nothing
to_cpu(x::Number)         = x
to_cpu(nt::NamedTuple)    = map(to_cpu, nt)
to_cpu(t::Tuple)          = map(to_cpu, t)

# ── NeuralIFGraph device transfer ─────────────────────────────────────────────

"""
    to_gpu(graph::NeuralIFGraph) -> NeuralIFGraph

Move all feature arrays and scatter indices in `graph` to the GPU.
Index arrays used only for host-side fancy-indexing (row_idx, col_idx,
lower_eidx) remain on the CPU — CUDA.jl supports CuArray[cpu_idx] natively.
"""
function to_gpu(graph::NeuralIFGraph)
    # All index arrays use Int32 on GPU (avoids 64-bit atomic ops in scatter kernels).
    _idx32(v) = CUDA.cu(Int32.(v))
    NeuralIFGraph(
        graph.n,
        _idx32(graph.row_idx),          # needed for NNlib.scatter aggregation on GPU
        _idx32(graph.col_idx),
        CUDA.cu(graph.edge_init),
        CUDA.cu(graph.inv_deg_edge),
        _idx32(graph.lower_row),        # needed for NNlib.scatter triangular apply
        _idx32(graph.lower_col),
        graph.lower_eidx,               # stays CPU: GPU fancy-indexing works with cpu idx
        CUDA.cu(graph.is_diag),
        CUDA.cu(graph.inv_deg_lower_row),
        CUDA.cu(graph.inv_deg_lower_col),
        CUDA.cu(graph.node_features),
        CUDA.CUSPARSE.CuSparseMatrixCSC(graph.A_scaled),
        CUDA.cu(graph.d_sqrt_inv),
    )
end

"""
    to_cpu(graph::NeuralIFGraph) -> NeuralIFGraph

Move all GPU arrays in `graph` back to the CPU.
"""
function to_cpu(graph::NeuralIFGraph)
    graph.A_scaled isa SparseMatrixCSC && return graph   # already on CPU, no copy needed
    _idx(v) = Vector{Int}(Array(v))
    NeuralIFGraph(
        graph.n,
        _idx(graph.row_idx),
        _idx(graph.col_idx),
        Array(graph.edge_init),
        Array(graph.inv_deg_edge),
        _idx(graph.lower_row),
        _idx(graph.lower_col),
        graph.lower_eidx,
        Array(graph.is_diag),
        Array(graph.inv_deg_lower_row),
        Array(graph.inv_deg_lower_col),
        Array(graph.node_features),
        _sparse_to_cpu(graph.A_scaled),
        Array(graph.d_sqrt_inv),
    )
end

# Convert CUSPARSE → SparseMatrixCSC (or pass through if already CPU sparse)
_sparse_to_cpu(A::SparseMatrixCSC) = A
_sparse_to_cpu(A) = SparseMatrixCSC(A)   # CuSparseMatrixCSC → SparseMatrixCSC

# ── Helper: check whether params NamedTuple lives on GPU ─────────────────────

"""
    _params_on_gpu(params) -> Bool

Return true if the first array found in `params` is a CuArray.
"""
_params_on_gpu(x::CUDA.CuArray)   = true
_params_on_gpu(x::AbstractArray)  = false
_params_on_gpu(::Nothing)         = false
_params_on_gpu(x::Number)         = false
_params_on_gpu(nt::NamedTuple)    = any(_params_on_gpu, values(nt))
_params_on_gpu(t::Tuple)          = any(_params_on_gpu, t)
