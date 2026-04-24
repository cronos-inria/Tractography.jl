# GPU example

This page shows a GPU workflow for tractography.
The implementation is backend-agnostic (CUDA, Metal), but the main example below uses CUDA.

To maximize throughput, we use `Float32`.

!!! danger
    GPU allocations are expensive. Pre-allocate arrays once and reuse them in-place.

```@example GPU
using Tractography
const TG = Tractography

# use the TMC as the streamline model
model = TMC(Δt = 0.125f0,
            foddata = FODData((@__DIR__) * "/../../examples/fod-FC.nii.gz"),
            cone = Cone(45f0),
            proba_min = 0.005f0,
            )
```

# Define the seeds

```julia
using CUDA
# number of streamlines
Nmc = 1024*400
# maximum number of steps for each streamline
Nt = 2000
# define the seeds
seeds = cu(zeros(6, Nmc));
seeds[1:3, :] .= [-13.75, 26.5, 8] .+ 0.1  .* randn(3, Nmc) .|> Float32 |> CuArray;
seeds[4, :] .= 1
tract_length = CuArray(zeros(UInt32, Nmc))
```

!!! details "GPU on Apple OSX"
    ```julia
    using Metal
    cu = MtlArray{Float32}
    # number of streamlines
    Nmc = 1024*400
    # maximum number of steps for each streamline
    Nt = 2000
    # define the seeds
    seeds = cu(zeros(6, Nmc));
    seeds[1:3, :] .= [-13.75, 26.5, 8] .+ 0.1  .* randn(3, Nmc) .|> Float32 |> MtlArray;
    seeds[4, :] .= 1
    tract_length = MtlArray(zeros(UInt32, Nmc))
    ```

    The rest of the workflow is unchanged.



```julia
streamlines_gpu = cu(zeros(Float32, 3, Nt, Nmc), unified = true)
```

# Define the computation cache

Because we often run multiple batches on the same `model`, precomputing the cache is recommended.

```julia
# we precompute the cache which is heavy otherwise each call to sample
# will recompute it
cache_g = TG.init(model, Probabilistic(); 
                  𝒯ₐ = CuArray,
                  n_sphere = 400);
```

# Compute the streamlines

The following takes 0.5s on a A100.

```julia
# this setup works for a GPU with 40GiB
# it yields 1e6/sec streamlines for Probabilistic
# and 2.2e6/sec streamlines for Deterministic
CUDA.@time TG.sample!(
            streamlines_gpu,
            tract_length,
            model,
            cache_g,
            Probabilistic(),
            seeds;
            gputhreads = 1024,
            );
```

!!! tip "batches"
    This can be called many times, for example after updating the seeds.


`streamlines_gpu` stays on device memory. To inspect the result on CPU, copy or wrap depending on your backend.

```julia
streamlines = @time unsafe_wrap(Array, streamlines_gpu);
```