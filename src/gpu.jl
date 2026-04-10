import KernelAbstractions as KA
import KernelAbstractions: @kernel, @index

"""
$(TYPEDSIGNATURES)

Create a cache for computing streamlines in batches. This is interesting for use of memory limited environments (e.g. on GPU).

!!! tip "Tip"
    Use it with `sample!`

# Arguments
- `alg` sampling algorithm, `Deterministic, Probabilistic, Diffusion`, etc.
- `n_sphere::Int = 400` number of points to discretize the sphere on which we evaluate the spherical harmonics.
- `𝒯ₐ = Array{𝒯}` specify the type of the arrays in the cache. If passed a GPU array type like `CuArray` for CUDA, the sampling occurs on the GPU. Leave it for computing on CPU.
"""
function init(model::TMC{𝒯},
                alg; 
                n_sphere = 400,
                𝒯ₐ = Array{𝒯},
                ) where 𝒯
    cache_cpu = _init(model, _get_alg(alg); n_sphere)
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

Sample the TMC `model` inplace by overwriting `result`. This requires very little memory and can be run indefinitely on the GPU for example.

# Arguments
- `streamlines` array with shape `3 x nt x Nmc`. `nt` is maximal length of each streamline. `Nmc` is the number of Monte-Carlo simulations to be performed.
- `streamlines_length` length of the streamlines
- `model::TMC` 
- `alg` sampling algorithm, `Deterministic, Probabilistic, Diffusion, etc`.
- `seeds` matrix of size `6 x Nmc` where `Nmc` is the number of Monte-Carlo simulations to be performed.

## Optional arguments
- `maxfod_start::Bool` for each locations, use direction provided by the argmax of the ODF.
- `reverse_direction::Bool` reverse initial direction.
- `nthreads::Int = 8` number of threads on CPU.
- `gputhreads::Int = 512` number of threads on GPU.
"""
function sample!(streamlines, 
                  streamlines_length,
                  model::TMC{𝒯}, 
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
        error("You did not pass a cone function to TMC!")
    end
        if saveat > 1
        error("This option is not yet available. Open an issue on the website if you want this feature.")
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
                            angles,
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
                            @Const(angles::AbstractArray{𝒯, 2}),
                            @Const(directions::AbstractMatrix{𝒯}),
                            @Const(cone::AbstractMatrix{𝒯}),
                            @Const(tf),
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

    total_proba = proba = proba0 = cone_c = zero(𝒯)
    conditioned_proba = proba_max = zero(𝒯)

    voxel_index₁ = voxel_index₂ = voxel_index₃ = Int32(0)

    if maxfod_start
        voxel_index₁, voxel_index₂, voxel_index₃ = get_voxel_index(tf, (x₁, x₂, x₃))
        ind_u = _device_argmax(fodf, voxel_index₁, voxel_index₂, voxel_index₃, n_angles)
        u₁ = directions[ind_u, 1]; u₂ = directions[ind_u, 2]; u₃ = directions[ind_u, 3]
    end

    if reverse_direction
        u₁ = -u₁
        u₂ = -u₂
        u₃ = -u₃
    end

    if reverse_direction || ~maxfod_start
        ind_u = _device_get_angle(directions, u₁, u₂, u₃, n_angles)
    end

    streamlines[1, 1, nₙₘ] = x₁
    streamlines[2, 1, nₙₘ] = x₂
    streamlines[3, 1, nₙₘ] = x₃

    for iₜ = 2:nₜ
        # x is in native space
        (voxel_index₁, voxel_index₂, voxel_index₃) = get_voxel_index(tf, (x₁, x₂, x₃))

        continue_tracking = continue_tracking && in_image(voxel_index₁, voxel_index₂, voxel_index₃, nx, ny, nz)
        t_length += continue_tracking

        if continue_tracking
            # we compute the probabilities associated to the odf
            total_proba = proba = zero(𝒯)
            conditioned_proba = proba_max = zero(𝒯)
            ind_max = 0
            for i in 1:n_angles # use of axes prevents from optimization, better use 1:n
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
            # save current index of angle
            ind_u0 = ind_u

            if conditioned_proba > proba_min / dΩ &&
                        conditioned_proba > proba_min * total_proba
                if alg isa DeterministicSampler
                    ind_u = ind_max
                else
                    # cumulative sampling distribution (Probabilistic)
                    # only if probabilities large enough
                    # t = _rand[nₙₘ, iₜ] * conditioned_proba
                    t = rand(𝒯) * conditioned_proba
                    proba0 = zero(𝒯) # it is >= 0 already! 
                    cw = zero(𝒯)
                    for nₐ = 1:n_angles
                        # compute proba
                        proba0 = fodf[nₐ, voxel_index₁, voxel_index₂, voxel_index₃]
                        cone_c = cone[nₐ, ind_u0]
                        cw += proba0 * cone_c
                        if cw >= t
                            ind_u = nₐ
                            break
                        end
                    end
                end
            else
                # we stop tracking then
                continue_tracking = false
                if ~save_full_streamline
                    streamlines[1, 2, nₙₘ] = x₁
                    streamlines[2, 2, nₙₘ] = x₂
                    streamlines[3, 2, nₙₘ] = x₃
                end
            end
        end

        if continue_tracking
            u₁ = directions[ind_u, 1]
            u₂ = directions[ind_u, 2]
            u₃ = directions[ind_u, 3]

            x₁ += Δt * u₁
            x₂ += Δt * u₂
            x₃ += Δt * u₃
        end

        if save_full_streamline
            streamlines[1, iₜ, nₙₘ] = x₁
            streamlines[2, iₜ, nₙₘ] = x₂
            streamlines[3, iₜ, nₙₘ] = x₃
        end 
    end
    streamlines_length[nₙₘ] = t_length
end

@inline function in_image(voxel_index₁::Integer, voxel_index₂::Integer, voxel_index₃::Integer, nx, ny, nz)
    return 1 <= voxel_index₁ <= nx &&
           1 <= voxel_index₂ <= ny &&
           1 <= voxel_index₃ <= nz
end

@inline function get_voxel(tf::Transform, x_native)
    # x_native is in native space, we want it in voxel space
    return transform_inv(tf, SA.SVector(x_native[1], x_native[2], x_native[3], 1))
end

@inline function get_voxel_index(x_voxel)
    # we use the hack unsafe_trunc instead of Int(round(...)) because Metal 
    # doesn't provide a device-side allocator.
    # `+ 1` is for Julia array indexing which starts at 1
    @inbounds voxel_index = (unsafe_trunc(UInt32, round(x_voxel[1], RoundNearest) + 1),
                             unsafe_trunc(UInt32, round(x_voxel[2], RoundNearest) + 1),
                             unsafe_trunc(UInt32, round(x_voxel[3], RoundNearest) + 1))
    return voxel_index
end

@inline function get_voxel_index(tf::Transform, x_native)
    x = get_voxel(tf, x_native)
    return get_voxel_index(x)
end

@inline function _device_argmax(fodf::AbstractArray{𝒯, 4}, voxel₁, voxel₂, voxel₃, n::UInt32) where {𝒯}
    _val_max = zero(𝒯)
    ind_u = UInt32(1)
    for ii = UInt32(1):n
        @inbounds val = fodf[ii, voxel₁, voxel₂, voxel₃]
        if val > _val_max
            _val_max = val
            ind_u = ii
        end
    end
    return ind_u
end

@inline function _device_get_angle(directions::AbstractMatrix{𝒯}, u1::𝒯, u2::𝒯, u3::𝒯, n::UInt32) where {𝒯}
    ind_u = UInt32(1); i = UInt32(2)
    @inbounds val0 = directions[1, 1] * u1 + directions[1, 2] * u2 + directions[1, 3] * u3
    for i = UInt32(2):n
        @inbounds val = directions[i, 1] * u1 + 
                        directions[i, 2] * u2 + 
                        directions[i, 3] * u3
        if val0 < val
            val0 = val
            ind_u = i
        end
    end
    return ind_u
end