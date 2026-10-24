# Exporting Tractography Results

After computing a tractogram, you can export streamlines to the TCK file format for use with other neuroimaging tools like MRtrix3, TrackVis, etc.

## Saving to TCK Format

The TCK format is a widely-used binary format for storing tractography streamlines. `Tractography.jl` provides two writers:

- [`save_streamlines`](@ref) uses the Python library `nibabel` via `PythonCall.jl`.
- [`save_streamlines_julia`](@ref) writes the TCK format natively in Julia, with no Python dependency.

### Native Julia writer (recommended)

```julia
using Tractography
const TG = Tractography

# ... model definition skipped
streamlines, tract_length = TG.sample(model, alg_tracking, seeds)

# Export to TCK file
TG.save_streamlines_julia("my_tractogram.tck", streamlines, tract_length)
```

### Python/nibabel writer

Ensure `PythonCall.jl` is installed and `nibabel` is available in its Python environment, then load `PythonCall` to enable [`save_streamlines`](@ref):

```julia
using Tractography, PythonCall
const TG = Tractography

streamlines, tract_length = TG.sample(model, alg_tracking, seeds)
TG.save_streamlines("my_tractogram.tck", streamlines, tract_length)
```

## Output Format

The saved TCK file contains:
- **Streamline coordinates**: 3D positions of points along each fiber tract, stored as little-endian `Float32`
- **Streamline lengths**: the number of valid points of each saved fiber

Fibers with two points or less are not saved. The file can be visualized and analyzed using standard tractography software such as MRtrix3.