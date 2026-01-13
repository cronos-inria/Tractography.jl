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