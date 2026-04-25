# Generating Seeds

Seeds are starting points for tractography streamlines. `Tractography.jl` provides two main methods for generating seeds:

## From FOD Data

Generate seeds directly from Fiber Orientation Distribution (FOD) data using [`Tractography.from_fod`](@ref). This method places seeds throughout the FOD volume, typically in voxels with significant fiber content.

```julia
using Tractography
const TG = Tractography

# Load FOD data
model = TG.Model(
    foddata = TG.FODData("path/to/fod.nii.gz"),
    # other arguments ...
)

# Generate 100 seeds from FOD
seeds = TG.from_fod(model, 100)
```

## From Mask

Generate seeds from a binary or labeled mask using [`Tractography.from_mask`](@ref). This method is useful when you want to restrict seeding to specific anatomical regions.

```julia
# Generate 100 seeds from a mask array
mask = # your mask array (3D binary or labeled volume)
seeds = TG.from_mask(model, mask, 100)
```

## Seed Format

Seeds are typically represented as a matrix where each column contains the (x, y, z, ux, uy, uz) coordinates of a seed point in voxel space.