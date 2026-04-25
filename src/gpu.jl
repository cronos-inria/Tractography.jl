import KernelAbstractions as KA
import KernelAbstractions: @kernel, @index

"""
$(TYPEDSIGNATURES)

Create a cache for computing streamlines in batches. This is useful for memory-limited environments (e.g. GPU).

!!! tip "Tip"
    Use it with `sample!`

# Arguments
- `alg` sampling algorithm, `Deterministic, Probabilistic, Diffusion`, etc.
- `n_sphere::Int = 400` number of points to discretize the sphere for spherical harmonics evaluation.
- `𝒯ₐ = Array{𝒯}` array type for the cache. Pass a GPU array type like `CuArray` to run on GPU; leave as `Array` for CPU.
"""
function init(model::Model{𝒯},
                alg; 
                n_sphere = 400,
                𝒯ₐ = Array{𝒯},
                ) where 𝒯
    cache_cpu = _init(model, _get_alg(alg), get_basis(model); n_sphere)
    # do not copy the array if the types are the same
    _is_on_cpu = cache_cpu.odf isa 𝒯ₐ

    ThreadedCache(
            _is_on_cpu ? cache_cpu.odf : 𝒯ₐ(cache_cpu.odf),
            𝒯ₐ(zeros(𝒯, 0,0,0,0)),
            𝒯ₐ(zeros(𝒯, 0,0,0,0)),
            _is_on_cpu ? cache_cpu.cone : 𝒯ₐ(cache_cpu.cone),
            𝒯ₐ(mapreduce(x->[x[1] x[2] x[3]], vcat, cache_cpu.directions)),
            𝒯ₐ(mapreduce(x->[x[1] x[2]], vcat, cache_cpu.angles)),
            nothing,
            cache_cpu.dΩ
    )
end

"""
$(TYPEDSIGNATURES)

Sample the `model` in place by overwriting `streamlines`. Uses minimal memory and can run indefinitely on GPU.

# Arguments
- `streamlines` array with shape `3 x nt x Nmc`. `nt` is the maximum length per streamline. `Nmc` is the number of Monte-Carlo simulations.
- `streamlines_length` lengths of the streamlines.
- `model::Model` model to sample from.
- `alg` sampling algorithm: `Deterministic`, `Probabilistic`, `Diffusion`, etc.
- `seeds` matrix of size `6 x Nmc` with positions (x,y,z) and directions (u,v,w).

## Optional arguments
- `maxfod_start::Bool` use the argmax direction of the ODF at each location.
- `reverse_direction::Bool` reverse the initial direction.
- `nthreads::Int = 8` number of CPU threads.
- `gputhreads::Int = 512` number of GPU threads.
"""
function sample!(streamlines, 
                  streamlines_length,
                  model::Model{𝒯}, 
                  cache::AbstractCache, 
                  alg,
                  seeds;
                  maxfod_start::Bool = false,
                  reverse_direction::Bool = false,
                  nthreads = 8,
                  gputhreads = 512,
                  nₜ = size(streamlines, 2),
                  saveat::Int = 1,
                  𝒯ₐ = Array) where {𝒯}
    _, nx, ny, nz = size(cache.odf)
    streamlines_length .= nₜ
    if isnothing(cache.cone)
        error("You did not pass a cone function to the model")
    end
    if saveat > 1
        error("This option is not yet available. Open an issue on the website if you want this feature.")
    end
    if !(get_evaluation(model) isa PreComputeAllFOD)
        error("Only the evaluation strategy PreComputeAllFOD is allowed for `Deterministic` and `Probabilistic`.")
    end
    # the following allows for type inference
    launch_kernel(nthreads;
                    streamlines,
                    streamlines_length,
                    alg,
                    seeds,
                    odf = cache.odf,
                    angles = cache.angles,
                    directions = cache.directions,
                    cone = cache.cone,
                    transform = model.foddata.transform,
                    maxfod_start,
                    reverse_direction,
                    proba_min = model.proba_min,
                    dΩ = cache.dΩ,
                    Δt = model.Δt,
                    nx, ny, nz, gputhreads, nₜ)
    return streamlines
end

function launch_kernel(nthreads = 8;
                        streamlines::AbstractArray{𝒯, 𝒩},
                        streamlines_length::AbstractVector{UInt32},
                        alg,
                        seeds::AbstractMatrix{𝒯},
                        odf::AbstractArray{𝒯, 4},
                        angles::AbstractMatrix{𝒯},
                        directions::AbstractMatrix{𝒯},
                        cone::AbstractMatrix{𝒯},
                        transform,
                        maxfod_start,
                        reverse_direction,
                        proba_min::𝒯,
                        dΩ::𝒯,
                        Δt::𝒯,
                        nx, ny, nz,
                        gputhreads = 512,
                        nₜ = size(streamlines, 2),
                        saveat::Int = 1,
                        ) where {𝒯, 𝒩}
    Nmc = size(seeds, 2)
    if size(seeds, 1) != 6 
        error("The initial positions must be passed as an 6 x N array.")
    end
    if (size(directions, 2) != 3) || (Nmc > size(streamlines, 3))
        error("The size of direction or the size of streamlines is too small!")
    end
    if size(streamlines, 3) != length(streamlines_length)
        error("You must pass an abstract Vector `streamlines_length` whose length matches the last dimension of `streamlines`")
    end
    @debug "" size(streamlines) nthreads gputhreads size(odf) nx ny nz alg

    # launch cpu / gpu kernel
    backend = KA.get_backend(seeds)
    _nthreads = backend isa KA.GPU ? gputhreads : nthreads
    kernel! = _sample_kernel!(backend, _nthreads)
    @time "kernel " kernel!(
                            streamlines, 
                            streamlines_length,
                            _get_alg(alg),
                            seeds,
                            odf,
                            directions,
                            cone,
                            transform,
                            Int32(nₜ),
                            maxfod_start,
                            reverse_direction,
                            proba_min,
                            dΩ,
                            Δt,
                            nx, ny, nz,
                            Val(~(alg isa Connectivity))
                            ;
                            ndrange = Nmc
                            )
    return streamlines
end

# this type annotation may help KA
KA.@kernel inbounds=true function _sample_kernel!(
                            streamlines::AbstractArray{𝒯, 3},
                            streamlines_length::AbstractArray{UInt32, 1},
                            @Const(alg::Union{Probabilistic, Deterministic}),
                            @Const(seeds::AbstractMatrix{𝒯}),
                            @Const(fodf::AbstractArray{𝒯, 4}),
                            @Const(directions::AbstractMatrix{𝒯}),
                            @Const(cone::AbstractMatrix{𝒯}),
                            @Const(transform),
                            @Const(nₜ::Int32),
                            @Const(maxfod_start::Bool),
                            @Const(reverse_direction::Bool),
                            @Const(proba_min::𝒯),
                            @Const(dΩ::𝒯),
                            @Const(Δt::𝒯),
                            @Const(nx), @Const(ny), @Const(nz),
                            ::Val{save_full_streamline}
                            ) where {𝒯, save_full_streamline}
    # index of the streamline being computed
    nₙₘ = @index(Global)

    x₁ = seeds[1, nₙₘ]; x₂ = seeds[2, nₙₘ]; x₃ = seeds[3, nₙₘ]
    u₁ = seeds[4, nₙₘ]; u₂ = seeds[5, nₙₘ]; u₃ = seeds[6, nₙₘ]

    n_angles = UInt32(size(fodf, 1))

    # current index of angle
    ind_u::UInt32 = 1
    ind_u0::UInt32 = 1
    ind_max::UInt32 = 0
    # streamline length
    t_length::UInt32 = 1

    inside_image::Bool = true
    continue_tracking::Bool = true

    total_proba = conditioned_proba = zero(𝒯)

    voxel_index₁ = voxel_index₂ = voxel_index₃ = Int32(0)
    precomputed_odf::Bool = true

    (;ind_u, u₁, u₂, u₃, voxel_index₁, voxel_index₂, voxel_index₃) = _init_streamline(
                                    maxfod_start, reverse_direction, precomputed_odf,
                                    transform, fodf, directions, n_angles,
                                    x₁, x₂, x₃, u₁, u₂, u₃)

    streamlines[1, 1, nₙₘ] = x₁; streamlines[2, 1, nₙₘ] = x₂; streamlines[3, 1, nₙₘ] = x₃

    for iₜ = 2:nₜ
        # x is in native space
        (voxel_index₁, voxel_index₂, voxel_index₃) = get_voxel_index(transform, (x₁, x₂, x₃))

        continue_tracking = continue_tracking && in_image(voxel_index₁, voxel_index₂, voxel_index₃, nx, ny, nz)
        t_length += continue_tracking

        if continue_tracking
            conditioned_proba, total_proba, ind_max = 
                            orientation_probabilities(alg,
                                                    fodf,
                                                    cone,
                                                    voxel_index₁, voxel_index₂, voxel_index₃,
                                                    ind_u,
                                                    n_angles)
            # save current index of angle
            ind_u0 = ind_u

            if conditioned_proba > proba_min / dΩ &&
                        conditioned_proba > proba_min * total_proba
                ind_u = next_orientation(alg,
                                         ind_u,
                                         ind_max,
                                         conditioned_proba,
                                         fodf,
                                         cone,
                                         voxel_index₁, voxel_index₂, voxel_index₃,
                                         ind_u0,
                                         n_angles)
            else
                # we stop tracking then
                continue_tracking = false
                if ~save_full_streamline
                    streamlines[1, 2, nₙₘ] = x₁; streamlines[2, 2, nₙₘ] = x₂; streamlines[3, 2, nₙₘ] = x₃
                end
            end
        end

        if continue_tracking
            u₁ = directions[ind_u, 1]; u₂ = directions[ind_u, 2]; u₃ = directions[ind_u, 3]
            x₁ += Δt * u₁; x₂ += Δt * u₂; x₃ += Δt * u₃
        end

        if save_full_streamline
            streamlines[1, iₜ, nₙₘ] = x₁; streamlines[2, iₜ, nₙₘ] = x₂; streamlines[3, iₜ, nₙₘ] = x₃
        end 
    end # for-loop
    streamlines_length[nₙₘ] = t_length
end
@inline function orientation_probabilities(alg,
                                          fodf::AbstractArray{𝒯, 4},
                                          cone::AbstractMatrix{𝒯},
                                          voxel_index₁, voxel_index₂, voxel_index₃,
                                          ind_u::UInt32,
                                          n_angles::UInt32) where {𝒯}
    total_proba = conditioned_proba = proba_max = zero(𝒯)
    ind_max = UInt32(0)
    for i = UInt32(1):n_angles # use of axes prevents from optimization, better use 1:n
        proba0 = fodf[i, voxel_index₁, voxel_index₂, voxel_index₃] # it is >= 0 already!
        cone_c = cone[i, ind_u]
        proba = proba0 * cone_c
        # keep track of conditional probabilities
        conditioned_proba += proba
        total_proba += proba0
        # we pre-compute this in case alg isa DeterministicSampler
        if alg isa DeterministicSampler
            if proba > proba_max
                proba_max = proba
                ind_max = i
            end
        end
    end
    return conditioned_proba, total_proba, ind_max
end

@inline next_orientation(::DeterministicSampler,
                                  ind_u::UInt32,
                                  ind_max::UInt32,
                                  conditioned_proba::𝒯,
                                  fodf::AbstractArray{𝒯, 4},
                                  cone::AbstractMatrix{𝒯},
                                  voxel_index₁, voxel_index₂, voxel_index₃,
                                  ind_u0::UInt32,
                                  n_angles::UInt32) where {𝒯} = ind_max

@inline function next_orientation(::Probabilistic,
                                  ind_u::UInt32,
                                  ind_max::UInt32,
                                  conditioned_proba::𝒯,
                                  fodf::AbstractArray{𝒯, 4},
                                  cone::AbstractMatrix{𝒯},
                                  voxel_index₁, voxel_index₂, voxel_index₃,
                                  ind_u0::UInt32,
                                  n_angles::UInt32) where {𝒯}
    # cumulative sampling distribution (Probabilistic)
    t = rand(𝒯) * conditioned_proba
    proba0 = zero(𝒯) # it is >= 0 already!
    cw = zero(𝒯)
    for nₐ = UInt32(1):n_angles
        # compute proba
        proba0 = fodf[nₐ, voxel_index₁, voxel_index₂, voxel_index₃]
        cone_c = cone[nₐ, ind_u0]
        cw += proba0 * cone_c
        if cw >= t
            ind_u = nₐ
            break
        end
    end
    return ind_u
end
