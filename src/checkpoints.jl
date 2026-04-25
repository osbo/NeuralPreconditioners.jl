# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint API: save and load NeuralIF params + config
#
# Uses Julia's stdlib Serialization module — no extra dependencies.
# Note: the serialized format is tied to the Julia version used to write it.
# For cross-version portability, re-serialize after upgrading Julia.
# ─────────────────────────────────────────────────────────────────────────────

using Serialization

"""
    save_neuralif(path, params, cfg)

Save NeuralIF parameters and config to `path` (conventionally `.jls`).

Parameters are moved to CPU before saving regardless of where they currently
live, so GPU-trained params can be saved and reloaded on any machine.

# Example
    save_neuralif("my_preconditioner.jls", ps, cfg)
    ps2, cfg2 = load_neuralif("my_preconditioner.jls")
"""
function save_neuralif(path::AbstractString, params, cfg::NeuralIFConfig)
    params_cpu = to_cpu(params)
    Serialization.serialize(path, (; params=params_cpu, cfg=cfg))
    return path
end

"""
    load_neuralif(path) -> (params, cfg)

Load NeuralIF parameters and config saved by `save_neuralif`.

Returns a CPU NamedTuple parameter tree and a `NeuralIFConfig`. Call
`to_gpu(params)` before training or inference on GPU.

# Example
    ps, cfg = load_neuralif("my_preconditioner.jls")
    ps_gpu  = to_gpu(ps)
    M = NeuralIFPreconditioner(A, ps_gpu, cfg)
"""
function load_neuralif(path::AbstractString)
    data = Serialization.deserialize(path)
    return data.params, data.cfg
end
