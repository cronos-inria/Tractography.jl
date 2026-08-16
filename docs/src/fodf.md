# Specification of fODF and their evaluation

```@contents
Pages = ["fodf.md"]
Depth = 3
```

We detail the various ways fODF can be expressed. They are specified by a basis and an [evaluation strategy](@ref sheval).

# Basis for expressing fODF

The `basis` is passed to the `FODData`. The evaluation strategy is passed to the model (e.g. a `Model`, `Diffusion`, etc.).

## 1. Basis of spherical harmonics (SPH)

This case corresponds to `basis = SphericalHarmonics()` which is the default. The set of fODFs can be passed in two ways.

!!! danger "Spherical harmonics (SPH) basis issues"
    SPH bases do not enforce positivity of the FODF, which is required for a proper probability distribution. This is mitigated by passing the FODF through a mollifier in `Model`, but this alters the probabilities. A second issue is the Gibbs phenomenon, which introduces oscillations near discontinuities in the spherical signal. All this make SPH a not very good basis for tractography.

### Provided by a file

The easiest way is to use the constructor of [`Tractography.FODData`](@ref) and pass the nii file path directly. Here is a simple dummy example

```julia
import Tractography as TG

foddata = TG.FODData("fods.nii.gz", normalize_it = false)
```

### Provided by an `Array`

You can also pass directly the SPH coefficients as an array of size `(nx, ny, nz, 6)` to the constructor of [`Tractography.FODData`](@ref). The 6 coefficients are the even spherical harmonics coefficients, ordered by increasing `l` and then `m`.

Here is a simple dummy example

```julia
import Tractography as TG

foddata = TG.FODData(rand(10,10,10,6), false)
```

## 2. Basis of Diffusion Tensors (DTI)

This case corresponds to `basis = DTI()`. A diffusion tensor is symmetric so it has 6 independent coefficients, stored in the order `v_tensor = (D11, D22, D33, D12, D13, D23)`, i.e. as the matrix

```julia
[v_tensor[1] v_tensor[4] v_tensor[5];
 v_tensor[4] v_tensor[2] v_tensor[6];
 v_tensor[5] v_tensor[6] v_tensor[3]] 
```

You can pass the data using a file as for the `SphericalHarmonics` basis, provided that you pass `basis = DTI()` to the constructor of [`Tractography.FODData`](@ref). You can also pass directly the coefficients as an array to the constructor. Here is a simple dummy example

```julia
import Tractography as TG

foddata = TG.FODData(rand(10,10,10,6), false; basis = DTI())
```

# Evaluation strategies

When sampling the streamlines, fODF have to be evaluated on the basis provided to `FODData`. The evaluation strategy is passed to a model (e.g. `Model`).

We provide two modes of evaluation
1. the fODF are precomputed for various directions on the sphere and stored in a cache. This corresponds to passing `evaluation_algo = PreComputeAllFOD()` to the model. The number of sampling points on sphere is passed to the function `TG.sample` or `TG.sample!`.
2. the fODF are evaluated on the fly during sampling of the streamlines. This corresponds to passing `evaluation_algo = DirectFOD()` to the model. Only `Diffusion` and `Transport` models are allowed for this mode.
