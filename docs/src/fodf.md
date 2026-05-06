# Specification of fODF and their evaluation

```@contents
Pages = ["fodf.md"]
Depth = 3
```

We detail the various ways fODF can be expressed. They are specified by a basis and an [evaluation strategy](@ref sheval).

The `basis` is passed to the `FODData`. The evolution strategy is passed to the model (e.g. a `Model`, `Diffusion`, etc.).

# Basis for expressing fODF

## 1. Basis of spherical harmonics (SPH)

This case corresponds to `basis = SphericalHarmonics()` which is the default. The set of fODFs can be passed in two ways.

!!! danger "Spherical harmonics (SPH) basis issues"
    SPH bases do not enforce positivity of the FODF, which is required for a proper probability distribution. This is mitigated by passing the FODF through a mollifier in `Model`, but this alters the probabilities. A second issue is the Gibbs phenomenon, which introduces oscillations near discontinuities in the spherical signal. All this make SPH a not very good basis for tractography.

### Provided by a file

The easiest way is to use the constructor of [`Tractography.FODData`](@ref) and pass the nii file path directly. Here is a simple dummy example

```julia
import Tractography as TG

foddata = TG.FODData("fods.nii.gz", false)
```

### Provided by an `Array`

You can also pass directly the SPH coefficients as an array of size `(nx, nx, nz, 6)` to the constructor of [`Tractography.FODData`](@ref). The matrix layout is

```julia
[v_tensor[1] v_tensor[4] v_tensor[5];
 v_tensor[4] v_tensor[2] v_tensor[6];
 v_tensor[5] v_tensor[6] v_tensor[3]] 
```

Here is a simple dummy example

```julia
import Tractography as TG

foddata = TG.FODData(rand(10,10,10,6), false)
```

## 2. Basis of Diffusion Tensors (DTI)

This case corresponds to `basis = DTI()`. You can pass the data using a file as for the `SphericalHarmonics` basis. You can also pass directly the coefficients as an array to the constructor of [`Tractography.FODData`](@ref). Here is a simple dummy example

```julia
import Tractography as TG

foddata = TG.FODData(rand(10,10,10,10), false)
```

# Evaluation strategies

When sampling the streamlines, fODF have to be evaluated on the basis provided to `FODData`. The evaluation strategy is passed to a model (e.g. `Model`).

We provide two modes of evaluation
1. the fODF are precomputed for various directions on the sphere and stored in a cache. This corresponds to passing `evaluation_algo = PreComputeAllFOD()` to the model. The number of sampling points on sphere is passed to the function `TG.sample` or `TG.sample!`.
2. the fODF are evaluated on the fly during sampling of the streamlines. This corresponds to passing `evaluation_algo = DirectFOD()` to the model. Only `Diffusion` and `Transport` models are allowed for this mode.
