# Exporting Tractography Results

After computing a tractogram, you can export streamlines to the TCK file format for use with other neuroimaging tools like MRtrix3, TrackVis, etc.

## Saving to TCK Format

The TCK format is a widely-used binary format for storing tractography streamlines. `Tractography.jl` provides a convenience function [`save_streamlines`](@ref) that uses the Python library `nibabel` via `PythonCall.jl` to export streamlines.

### Prerequisites

Ensure you have `PythonCall.jl` installed and `nibabel` available in your Python environment:

```julia
using Pkg
Pkg.add("PythonCall")
```

Then make sure `nibabel` is available in the Python environment used by `PythonCall` (e.g. `pip install nibabel` in that environment, or `CondaPkg.add("nibabel")` from Julia).

### Basic Example

```julia
using Tractography
using PythonCall
const TG = Tractography

# ... model definition skipped
streamlines, tract_length = TG.sample(model, alg_tracking, seeds)

# Export to TCK file
TG.save_streamlines("my_tractogram.tck", streamlines, tract_length)
```

## Output Format

The saved TCK file contains:
- **Streamline coordinates**: 3D positions of points along each fiber tract

The file can be visualized and analyzed using standard tractography software.