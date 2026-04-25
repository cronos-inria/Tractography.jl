# Plotting 

Plotting functionality is provided through `Makie.jl` recipes. By loading one of the Makie backends (`CairoMakie`, `GLMakie`, or `WGLMakie`), you can use custom functions to visualize FOD (Fiber Orientation Distribution) data.

This tutorial uses `CairoMakie` for static plots, but `GLMakie` is recommended for interactive 3D visualization.

## Setup

Let's start by loading FOD data from a NIfTI file:

```@example PLOTTING
using Tractography
const TG = Tractography

model = TG.Model(Δt = 0.125f0,
            foddata = TG.FODData((@__DIR__) * "/../../examples/fod-FC.nii.gz"),
            cone = TG.Cone(15f0),
            proba_min = 0.005f0,
            )
```

## Plotting FODFs

To visualize Fiber Orientation Distribution Functions (FODFs), use the `plot_fod!` function. See the [`plot_fod!`](@ref) documentation for details on available keyword arguments.

The following example plots FODFs for a specific region of interest:

```@example PLOTTING
using CairoMakie
f, sc = TG.plot_fod(model; n_sphere = 1500, radius = 1., I = 29:31, J = 30:31, K = 2:2);
cam3d = Makie.cameracontrols(sc) # hide
cam3d.eyeposition[] = Vec3f(96.91587149289712, 85.36944962225634, -85.12319118529459)  # hide
cam3d.lookat[] = Vec3f(95.90360332812566, 85.36944962225634, 2.943478780350152)  # hide
cam3d.upvector[] = Vec3f(0.9999339466437674, 0.0, 0.01149357861675617)  # hide
Makie.zoom!(sc.scene, cam3d, 0.10f0)  # hide
f
```

## Plotting Streamlines

To visualize tractography streamlines, use the `plot_streamlines!` function. See the [`plot_streamlines!`](@ref) documentation for details on available keyword arguments.

The following example creates synthetic streamlines and plots them alongside FODFs:

```@example PLOTTING
# Generate synthetic streamlines for demonstration
streamlines = zeros(Float32, 6, 100, 20)
for n = 1:20
    v0 = rand(3) * 15  # Random starting position
    for nt = 1:100
        # Create curved trajectories
        streamlines[1:3, nt, n] .= v0 .+ nt .* [1, 1 + 0.1*rand(), 0]
    end
end

# Create a combined visualization of streamlines and FODFs
f = Figure(backgroundcolor = :white)
lscene = LScene(f[1, 1])
TG.plot_streamlines!(lscene.scene, streamlines[1:3, 1:1:end, :])
TG.plot_fod!(lscene, model; n_sphere = 1500, radius = 1.3, st = 4)
f
```