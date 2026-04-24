# 🚀 Get started with with Tractography.jl
```@contents
Pages = ["getstart.md"]
Depth = 3
```

This tutorial will introduce you to the functionalities for computing streamlines.

# Basic use

In this example, we sample `Nmc` streamlines from a `TMC` model.

```@example GS
using Tractography
const TG = Tractography

model = TMC(Δt = 0.125f0,
            foddata = FODData((@__DIR__) * "/../../examples/fod-FC.nii.gz"),
            )
Nmc = 10
seeds = rand(Float32, 6, Nmc)
alg = Probabilistic()
streamlines, tract_length = sample(model, alg, seeds);
size(streamlines)
```

## Step 1: Define a TMC

We define a Tractography Markov Chain (TMC) model as follows:

```@example GS
model = TMC(Δt = 0.125f0,
            foddata = FODData((@__DIR__) * "/../../examples/fod-FC.nii.gz"),
            )
```

## Step 2: Define the seeds

`seeds` is a `6 x Nmc` matrix. Each column stores:

- `seeds[1:3, i]`: initial position `(x, y, z)` in native space
- `seeds[4:6, i]`: initial direction `(u1, u2, u3)`

```@example GS
Nmc = 10 # Monte Carlo sample
seeds = rand(Float32, 6, Nmc)
```

## Step 3: Choose a sampling algorithm

```@example GS
alg = Probabilistic()
```

## Step 4: Sample the streamlines
```@example GS
streamlines, tract_length = sample(model, alg, seeds);
```

# Optimal use

When computing multiple batches for the same model, it is more efficient to precompute a cache once and reuse it.

```@example GS
model = TMC(Δt = 0.125f0,
            foddata = FODData((@__DIR__) * "/../../examples/fod-FC.nii.gz"),
            )
Nmc = 10
seeds = rand(Float32, 6, Nmc)
streamlines = zeros(Float32, 6, 20, Nmc)
tract_length = zeros(UInt32, Nmc)
alg = Probabilistic()
cache = TG.init(model, alg)
# this can be called repeatedly after updating seeds for example
TG.sample!(streamlines, tract_length, model, cache, alg, seeds);
size(streamlines)
```
