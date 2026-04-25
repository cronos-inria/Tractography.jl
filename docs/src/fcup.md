# FCUP

```@contents
Pages = ["fcup.md"]
Depth = 3
```

This is a more advanced tutorial because we want to show how to apply a mask.


# Define the Model

```@example FCUP
import Tractography as TG

model = TG.Model(Δt = 0.125f0,
            foddata = TG.FODData((@__DIR__) * "/../../examples/fod-FC.nii.gz"),
            cone = TG.Cone(45),
            proba_min = 0.015f0,
            )
```

Just for fun, we plot the FODs of the model.

```@example FCUP
using CairoMakie

f, sc = TG.plot_fod(model; n_sphere = 1500, radius = 0.3, st = 2);
cam3d = Makie.cameracontrols(sc)
cam3d.eyeposition[] = Vec3f(85, 95, -28)
cam3d.lookat[] = Vec3f(84, 95, 59)
rotate_cam!(sc.scene, 0, 0, -pi/2)
f
```

# Define the seeds

We next apply a mask on the boundary of which the streamlines stop.

```@example FCUP
using NIfTI
mask = NIfTI.niread((@__DIR__) * "/../../examples/wm-FC.nii.gz");
TG._apply_mask!(model, mask);
```

We compute `Nmc` streamlines, hence we need `Nmc` seeds

```@example FCUP
Nmc = 100_000
seeds = TG.from_fod(model, Nmc; maxfod_start = true)
```

# Compute the streamlines

```@example FCUP
streamlines, tract_length = TG.sample(model, TG.Deterministic(), seeds; nt = 1000);
println("Dimension of computed streamlines = ", size(streamlines))
```

# plot the streamlines

```@example FCUP
f, scene = @time TG.plot_fod(model; n_sphere = 500, radius = 0.3, st = 1);
ind_st = findall(tract_length .> 60)
TG.plot_streamlines!(scene, streamlines[:, :, ind_st[1:10:end]])
f
```

We can also add the seeds
```@example FCUP
scatter!(scene, seeds[1:3, ind_st[1:10:end]], color = :white)
f
```

# Compute the connections

When computing structural connectivity, we don't need to record the entire streamline but only its extremities.

```@example FCUP
streamlines, tract_length = TG.sample(model, TG.Connectivity(TG.Deterministic()), seeds; nt = 1000);
println("Dimension of computed streamlines = ", size(streamlines))
```