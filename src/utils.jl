####################################################################################################
macro time_debug(msg, ex)
    quote
        local ret = @timed $(esc(ex))
        local _msg = $(esc(msg))
        local _msg_str = _msg === nothing ? _msg : string(_msg)
        time = strip(sprint(Base.time_print, ret.time*1e9, ret.gcstats.allocd, ret.gcstats.total_time, Base.gc_alloc_count(ret.gcstats)))
        @debug _msg_str * " " * time
        ret.value
    end
end
####################################################################################################
"""
$(TYPEDSIGNATURES)

Transform spherical to cartesian coordinates, the chart on the sphere minus the north/south poles.

Its coordinates are (sin(θ)cos(ϕ), sin(θ)sin(ϕ), cos(θ)).

Recall that θ ∈ [0, π] and ϕ ∈ [0, 2π].
"""
@inline function spherical_to_euclidean(θ, ϕ)
    st, ct = sincos(θ)
    sp, cp = sincos(ϕ)
    x = st * cp
    y = st * sp
    z = ct
    return x, y, z 
end

"""
$(TYPEDSIGNATURES)

Transform cartesian to spherical coordinates. Assume that the vector has norm one.
Return (θ, ϕ) where θ ∈ [0, π] and ϕ ∈ [-π, π].
"""
@inline function euclidean_to_spherical(x, y, z)
    t = acos(z)
    p = atan(y, x)
    return t, p
end
####################################################################################################
"""
$(TYPEDSIGNATURES)

Returns points on the sphere in spherical coordinates which are approximately uniformly distributed. Each point (θ, ϕ) is such that θ ∈ [0, π] and ϕ ∈ [0, 2π].
"""
function fibonacci_sampling(N, 𝒯::DataType = Float64)
    out = Vector{Tuple{𝒯, 𝒯}}(undef, N+1)
    ϕ = (1 + √5)/2
    I = 0:N .+ 1/2
    r = 2π / ϕ
    for (n, i) in enumerate(I)
        out[n] = ( 𝒯(acos(1 - 2i/N)), 𝒯(mod(r * i, 2π)) )
    end
    return out
end
####################################################################################################
@inline softplus(x, k = 1) = ifelse(k * x < 30, log1p(exp(k * x)) / k, x) # avoid Inf for x large
@inline ∂softplus(x, k = 1) = 1 / (1 + exp(-k * x))
####################################################################################################
@inline function in_image(voxel_index₁::Integer, voxel_index₂::Integer, voxel_index₃::Integer, nx, ny, nz)
    return 1 <= voxel_index₁ <= nx &&
           1 <= voxel_index₂ <= ny &&
           1 <= voxel_index₃ <= nz
end

@inline function get_voxel_index(x_voxel)
    @inbounds voxel_index = (unsafe_trunc(UInt32, round(x_voxel[1], RoundNearest) + 1),
                             unsafe_trunc(UInt32, round(x_voxel[2], RoundNearest) + 1),
                             unsafe_trunc(UInt32, round(x_voxel[3], RoundNearest) + 1))
    return voxel_index
end

@inline function get_voxel_index(tf::Transform, x_native)
    x = transform_inv(tf, SA.SVector(x_native[1], x_native[2], x_native[3], 1))
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

@inline function _init_streamline(
                                    maxfod_start::Bool,
                                    reverse_direction::Bool,
                                    precomputed_odf::Bool,
                                    tf,
                                    fodf::AbstractArray{𝒯, 4},
                                    directions::AbstractMatrix{𝒯},
                                    n_angles::UInt32,
                                    x₁, x₂, x₃,
                                    u₁, u₂, u₃
                                ) where {𝒯}
    voxel_index₁ = voxel_index₂ = voxel_index₃ = Int32(0)
    ind_u = UInt32(1)

    if maxfod_start && precomputed_odf
        voxel_index₁, voxel_index₂, voxel_index₃ = get_voxel_index(tf, (x₁, x₂, x₃))
        ind_u = _device_argmax(fodf, voxel_index₁, voxel_index₂, voxel_index₃, n_angles)
        u₁ = directions[ind_u, 1]
        u₂ = directions[ind_u, 2]
        u₃ = directions[ind_u, 3]
    end

    if reverse_direction
        u₁ = -u₁
        u₂ = -u₂
        u₃ = -u₃
    end

    if precomputed_odf && (reverse_direction || !maxfod_start)
        ind_u = _device_get_angle(directions, u₁, u₂, u₃, n_angles)
    end

    return (; ind_u, u₁, u₂, u₃, voxel_index₁, voxel_index₂, voxel_index₃)
end