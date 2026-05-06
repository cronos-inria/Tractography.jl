import ForwardDiff

"""
$(TYPEDSIGNATURES)

Get the FOD vector length corresponding to a given lmax.
This length is the result of

```
n = 0
for l = 0:2:lmax, m = -l:l
    n += 1
end
n
# gives sequence 1  6  15  28  45  66  91  120
# for lmax       0  2   4   6  8   10  12  14  
```
"""
get_fod_length(lmax) = div(lmax^2, 2) + div(3 * lmax, 2) + 1

"""
$(TYPEDSIGNATURES)

Get the lmax from the ODF length, ie `size(data, 4)`. The is the inverse mapping of `get_fod_length`.
"""
get_lmax_from_fod_length(n) = Int(-3/2 + sqrt(1 + 8*n)/2)

"""
$(TYPEDSIGNATURES)

Evaluate the θ derivative of real orthonormal spherical harmonics.

Based on `FastTransforms.sphevaluate` and `ForwardDiff.jl`
"""
function ∂θro_sh(θ, ϕ, l, m, outer_f = identity)
    ForwardDiff.derivative(z -> outer_f(ro_sh(z, ϕ, l, m)), θ)
end

"""
$(TYPEDSIGNATURES)

Evaluate the ϕ derivative of real orthonormal spherical harmonics.

Based on `FastTransforms.sphevaluate` and `ForwardDiff.jl`
"""
function ∂ϕro_sh(θ, ϕ, l, m, outer_f = identity)
    ForwardDiff.derivative(z -> outer_f(ro_sh(θ, z, l, m)), ϕ)
end

"""
$(TYPEDSIGNATURES)

Evaluate the real orthonormal spherical harmonics.

Recall that ∫ ODF = c₀₀ √4π

Based on `FastTransforms.sphevaluate`. You can also check the [url](https://juliaapproximation.github.io/FastTransforms.jl/dev/#FastTransforms.sphevaluate)
"""
function ro_sh(θ, ϕ, l, m)
    FastTransforms.sphevaluate(θ, ϕ, l, m)
end

"""
$(TYPEDSIGNATURES)

Evaluate the real orthonormal spherical harmonics Yₗₘ(θ, ϕ) for `(θ, Φ) ∈ angles` for l ∈ [0, lₘₐₓ] and m ∈ [-l, l]. 

It returns a 2d array `Yₗₘ[(θ, ϕ), index]`.

It is mainly used to cache the harmonics Yₗₘ for later use (evaluation of FOD).

# Arguments
- `angles` a vector of tuples
- `lmax::Int` maximum l for harmonics.
- `der`: if `der=0` return  `ro_sh`. If `der=1` return `∂θro_sh` otherwise return `∂ϕro_sh`. 

> The storage convention is explained in https://mrtrix.readthedocs.io/en/dev/concepts/spherical_harmonics.html#storage-conventions.
"""
function get_vector_of_sh(angles::AbstractVector{Tuple{𝒯, 𝒯}}, lmax, der::Int = 0; outer_f = identity) where 𝒯
    odf_length = get_fod_length(lmax)
    Yₗₘ = zeros(𝒯, length(angles), odf_length)
    for (i, (θ, ϕ) ) in pairs(angles)
        n = 1
        for l = 0:2:lmax, m = -l:l
            if der == 0
                y = ro_sh(θ, ϕ, l, m)
            elseif der == 1
                y = ∂θro_sh(θ, ϕ, l, m, outer_f)
            else
                y = ∂ϕro_sh(θ, ϕ, l, m, outer_f)
            end
            Yₗₘ[i, n] = y * (-1)^m
            n += 1
        end
    end
    Yₗₘ
end

# vector with expression for ∂Ylm∂ϕ
const YLM_DP_RHS = Any[
    :(0f0),

    :(0.54627421529f0 * st2 * 2f0 * c2p),
    :(-1.0925484305920792f0 * stct * cp),
    :(0f0),
    :(1.0925484305920792f0 * stct * sp),
    :(-0.54627421529f0     * st2 * 2f0 * s2p),

    :(0.62583573544f0  * st4 * 4f0 * c4p),
    :(-1.77013076978f0 * st3 * ct * 3f0 * c3p),
    :(0.47308734787f0  * st2 * (7f0 * ct2 - 1f0) * 2f0 * c2p),
    :(-0.66904654355f0 * stct * (7f0 * ct2 - 3f0) * cp),
    :(0f0),
    :(0.66904654355f0  * stct * (7f0 * ct2 - 3f0) * sp),
    :(-0.47308734787f0 * st2 * (7f0 * ct2 - 1f0) * 2f0 * s2p),
    :(1.77013076978f0  * st3 * ct * 3f0 * s3p),
    :(-0.62583573544f0 * st4 * 4f0 * s4p),

    :(0.68318410519f0  * st6 * 6f0 * c6p),
    :(-2.36661916223f0 * st5 * ct * 5f0 * c5p),
    :(0.50456490072f0  * st4 * (11f0 * ct2 - 1f0) * 4f0 * c4p),
    :(-0.92120525951f0 * st3 * (11f0 * ct3 - 3f0 * ct) * 3f0 * c3p),
    :(0.46060262975f0  * st2 * (33f0 * ct4 - 18f0 * ct2 + 1f0) * 2f0 * c2p),
    :(-0.58262136251f0 * st * (33f0 * ct5 - 30f0 * ct3 + 5f0 * ct) * cp),
    :(0f0),
    :(0.58262136251f0  * st * (33f0 * ct5 - 30f0 * ct3 + 5f0 * ct) * sp),
    :(-0.46060262975f0 * st2 * (33f0 * ct4 - 18f0 * ct2 + 1f0) * 2f0 * s2p),
    :(0.92120525951f0  * st3 * (11f0 * ct3 - 3f0 * ct) * 3f0 * s3p),
    :(-0.50456490072f0 * st4 * (11f0 * ct2 - 1f0) * 4f0 * s4p),
    :(2.36661916223f0  * st5 * ct * 5f0 * s5p),
    :(-0.68318410519f0 * st6 * 6f0 * s6p),

    :(0.72892666017f0  * st8 * 8f0 * c8p),
    :(-2.9157066407f0  * st7 * ct * 7f0 * c7p),
    :(0.53233276606f0  * st6 * (15f0 * ct2 - 1f0) * 6f0 * c6p),
    :(-3.4499106221f0  * st5 * (5f0 * ct3 - ct) * 5f0 * c5p),
    :(0.47841652475f0  * st4 * (65f0 * ct4 - 26f0 * ct2 + 1f0) * 4f0 * c4p),
    :(-1.2352661553f0  * st3 * (39f0 * ct5 - 26f0 * ct3 + 3f0 * ct) * 3f0 * c3p),
    :(0.45615225843f0  * st2 * (143f0 * ct6 - 143f0 * ct4 + 33f0 * ct2 - 1f0) * 2f0 * c2p),
    :(-0.10904124589f0 * st * (715f0 * ct7 - 1001f0 * ct5 + 385f0 * ct3 - 35f0 * ct) * cp),
    :(0f0),
    :(0.10904124589f0  * st * (715f0 * ct7 - 1001f0 * ct5 + 385f0 * ct3 - 35f0 * ct) * sp),
    :(-0.45615225843f0 * st2 * (143f0 * ct6 - 143f0 * ct4 + 33f0 * ct2 - 1f0) * 2f0 * s2p),
    :(1.2352661553f0   * st3 * (39f0 * ct5 - 26f0 * ct3 + 3f0 * ct) * 3f0 * s3p),
    :(-0.47841652475f0 * st4 * (65f0 * ct4 - 26f0 * ct2 + 1f0) * 4f0 * s4p),
    :(3.4499106221f0   * st5 * (5f0 * ct3 - ct) * 5f0 * s5p),
    :(-0.53233276606f0 * st6 * (15f0 * ct2 - 1f0) * 6f0 * s6p),
    :(2.9157066407f0   * st7 * ct * 7f0 * s7p),
    :(-0.72892666017f0 * st8 * 8f0 * s8p),
]

# Symbolic transform for division by st (sin(θ))
const _DIV_ST_RULES = Dict{Symbol,Any}(
    :st   => :(one(𝒯)),
    :st2  => :st,
    :st3  => :st2,
    :st4  => :st3,
    :st5  => :st4,
    :st6  => :st5,
    :st7  => :st6,
    :st8  => :st7,
    :stct => :ct,
)

function _divide_by_st(ex)
    if ex isa Symbol
        return get(_DIV_ST_RULES, ex, ex)
    elseif ex isa Expr
        return Expr(ex.head, map(_divide_by_st, ex.args)...)
    else
        return ex
    end
end

function _make_ylm_dp_block(acc::Symbol; divide_by_st::Bool)
    rhs_list = divide_by_st ? map(_divide_by_st, YLM_DP_RHS) : YLM_DP_RHS

    stmts = Any[:(n = 1)]
    for rhs in rhs_list
        push!(stmts, quote
            $acc += $(rhs) * V[n]
            n += 1
        end)
    end
    return Expr(:block, stmts...)
end

for DIV in (false, true)
    body = _make_ylm_dp_block(:ylm_dp; divide_by_st = DIV)

    @eval @inline function _accum_ylm_dp(::Val{$DIV},
                                         V::AbstractVector{𝒯},
                                         st::𝒯, ct::𝒯,
                                         st2::𝒯, st3::𝒯, st4::𝒯, st5::𝒯, st6::𝒯, st7::𝒯, st8::𝒯,
                                         ct2::𝒯, ct3::𝒯, ct4::𝒯, ct5::𝒯, ct6::𝒯, ct7::𝒯, ct8::𝒯,
                                         stct::𝒯,
                                         sp::𝒯, cp::𝒯,
                                         s2p::𝒯, c2p::𝒯,
                                         s3p::𝒯, c3p::𝒯,
                                         s4p::𝒯, c4p::𝒯,
                                         s5p::𝒯, c5p::𝒯,
                                         s6p::𝒯, c6p::𝒯,
                                         s7p::𝒯, c7p::𝒯,
                                         s8p::𝒯, c8p::𝒯,
                                        ) where {𝒯}
        ylm_dp = zero(𝒯)
        @inbounds begin
            $body
        end
        return ylm_dp
    end
end

"""
$(TYPEDSIGNATURES)

Computes S = ∑Vₗₘ⋅Yₗₘ(θ, ϕ) and its partial derivatives ∂ϕS and ∂θS.

Returns (S, ∂ϕS, ∂θS).

!!! warning "Limitation"
    This is currently limited to `lmax<=8`
"""
@inline ishtmtx_dot(phi::𝒯, theta::𝒯, V::AbstractVector{𝒯}) where {𝒯} =
    _ishtmtx_dot_impl(Val(false), phi, theta, V)

"""
$(TYPEDSIGNATURES)

Computes S = ∑Vₗₘ⋅Yₗₘ(θ, ϕ) and its partial derivatives ∂ϕS and ∂θS.

Returns (S, ∂ϕS/sin(θ), ∂θS).

!!! warning "Limitation"
    This is currently limited to `lmax<=8`

!!! danger "Internal function"
    The derivative ∂ϕS is returned divided by sin(θ). This is made possible because we only consider spherical harmonics with even l.
"""
@inline ishtmtx_dot_divst(phi::𝒯, theta::𝒯, V::AbstractVector{𝒯}) where {𝒯} =
    _ishtmtx_dot_impl(Val(true), phi, theta, V)

function _ishtmtx_dot_impl(::Val{DIV},
                           phi::𝒯,
                           theta::𝒯,
                           V::AbstractVector{𝒯},
                          ) where {DIV,𝒯}

    st, ct = sincos(theta)
    sp, cp = sincos(phi)

    s, c = sp, cp

    s, c = s * cp + c * sp, c * cp - s * sp
    s2p, c2p = s, c

    s, c = s * cp + c * sp, c * cp - s * sp
    s3p, c3p = s, c

    s, c = s * cp + c * sp, c * cp - s * sp
    s4p, c4p = s, c

    s, c = s * cp + c * sp, c * cp - s * sp
    s5p, c5p = s, c

    s, c = s * cp + c * sp, c * cp - s * sp
    s6p, c6p = s, c

    s, c = s * cp + c * sp, c * cp - s * sp
    s7p, c7p = s, c

    s, c = s * cp + c * sp, c * cp - s * sp
    s8p, c8p = s, c

    st2 = st * st
    st3 = st2 * st
    st4 = st3 * st
    st5 = st4 * st
    st6 = st5 * st
    st7 = st6 * st
    st8 = st7 * st

    ct2 = ct * ct
    ct3 = ct2 * ct
    ct4 = ct3 * ct
    ct5 = ct4 * ct
    ct6 = ct5 * ct
    ct7 = ct6 * ct
    ct8 = ct7 * ct

    stct = st * ct

    ylm    = zero(𝒯)
    ylm_dt = zero(𝒯)

    @inbounds begin
        # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
        # https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
        # what is the normalization?

        # --- Ylm values (indices +1 vs C) ---
        n::UInt32 = 1
        ylm += (𝒯(1/sqrt(4pi))) * V[n]; n += 1

        # Real spherical harmonics
        # 1/4*sqrt(15/pi) = 0.5462742152960396
        ylm += (0.54627421529f0 * st2 * s2p) * V[n]; n += 1
        # 1/2*sqrt(15/2/pi) = 0.7725484040463791
        ylm += (-1.0925484305920792f0 * stct * sp) * V[n]; n += 1
        ylm += (0.31539156525252005f0 * (3 * ct2 - 1)) * V[n]; n += 1
        ylm += (-1.0925484305920792f0 * stct * cp) * V[n]; n += 1
        ylm += (0.54627421529f0 * st2 * c2p) * V[n]; n += 1

        ylm += (0.62583573544f0  * st4 * s4p) * V[n]; n += 1
        ylm += (-1.77013076978f0 * st3 * ct * s3p) * V[n]; n += 1
        ylm += (0.47308734787f0  * st2 * (7 * ct2 - 1) * s2p) * V[n]; n += 1
        ylm += (-0.66904654355f0 * st  * (7 * ct2 - 3) * ct * sp) * V[n]; n += 1
        ylm += (0.10578554691520431f0  * (35 * ct2 * ct2 - 30 * ct2 + 3)) * V[n]; n += 1
        ylm += (-0.66904654355f0 * st  * (7 * ct2 - 3) * ct * cp) * V[n]; n += 1
        ylm += (0.47308734787f0  * st2 * (7 * ct2 - 1) * c2p) * V[n]; n += 1
        ylm += (-1.77013076978f0 * st3 * ct * c3p) * V[n]; n += 1
        ylm += (0.62583573544f0  * st4 * c4p) * V[n]; n += 1

        ylm += (0.68318410519f0  * st6 * s6p) * V[n]; n += 1
        ylm += (-2.36661916223f0 * st5 * ct * s5p) * V[n]; n += 1
        ylm += (0.50456490072f0  * st4 * (11 * ct2 - 1) * s4p) * V[n]; n += 1
        ylm += (-0.92120525951f0 * st3 * (11 * ct3 - 3 * ct) * s3p) * V[n]; n += 1
        ylm += (0.46060262975f0  * st2 * (33 * ct4 - 18 * ct2 + 1) * s2p) * V[n]; n += 1
        ylm += (-0.58262136251f0 * st * (33 * ct5 - 30 * ct3 + 5 * ct) * sp) * V[n]; n += 1
        ylm += (0.06356920226f0  * (231 * ct6 - 315 * ct4 + 105 * ct2 - 5)) * V[n]; n += 1
        ylm += (-0.58262136251f0 * st * (33 * ct5 - 30 * ct3 + 5 * ct) * cp) * V[n]; n += 1
        ylm += (0.46060262975f0  * st2 * (33 * ct4 - 18 * ct2 + 1) * c2p) * V[n]; n += 1
        ylm += (-0.92120525951f0 * st3 * (11 * ct3 - 3 * ct) * c3p) * V[n]; n += 1
        ylm += (0.50456490072f0  * st4 * (11 * ct2 - 1) * c4p) * V[n]; n += 1
        ylm += (-2.36661916223f0 * st5 * ct * c5p) * V[n]; n += 1
        ylm += (0.68318410519f0  * st6 * c6p) * V[n]; n += 1

        ylm += (0.72892666017f0 * st8 * s8p) * V[n]; n += 1
        ylm += (-2.9157066407f0 * st7 * ct * s7p) * V[n]; n += 1
        ylm += (0.53233276606f0 * st6 * (15f0 * ct2 - 1f0) * s6p) * V[n]; n += 1
        ylm += (-3.4499106221f0 * st5 * (5f0 * ct3 - ct) * s5p) * V[n]; n += 1
        ylm += (0.47841652475f0 * st4 * (65f0 * ct4 - 26f0 * ct2 + 1f0) * s4p) * V[n]; n += 1
        ylm += (-1.2352661553f0 * st3 * (39f0 * ct5 - 26f0 * ct3 + 3f0 * ct) * s3p) * V[n]; n += 1
        ylm += (0.45615225843f0 * st2 * (143f0 * ct6 - 143f0 * ct4 + 33f0 * ct2 - 1f0) * s2p) * V[n]; n += 1
        ylm += (-0.10904124589f0 * st * (715f0 * ct7 - 1001f0 * ct5 + 385f0 * ct3 - 35f0 * ct) * sp) * V[n]; n += 1
        ylm += (0.00908677049f0 * (6435f0 * ct8 - 12012f0 * ct6 + 6930f0 * ct4 - 1260f0 * ct2 + 35f0)) * V[n]; n += 1
        ylm += (-0.10904124589f0 * st * (715f0 * ct7 - 1001f0 * ct5 + 385f0 * ct3 - 35f0 * ct) * cp) * V[n]; n += 1
        ylm += (0.45615225843f0 * st2 * (143f0 * ct6 - 143f0 * ct4 + 33f0 * ct2 - 1f0) * c2p) * V[n]; n += 1
        ylm += (-1.2352661553f0 * st3 * (39f0 * ct5 - 26f0 * ct3 + 3f0 * ct) * c3p) * V[n]; n += 1
        ylm += (0.47841652475f0 * st4 * (65f0 * ct4 - 26f0 * ct2 + 1f0) * c4p) * V[n]; n += 1
        ylm += (-3.4499106221f0 * st5 * (5f0 * ct3 - ct) * c5p) * V[n]; n += 1
        ylm += (0.53233276606f0 * st6 * (15f0 * ct2 - 1f0) * c6p) * V[n]; n += 1
        ylm += (-2.9157066407f0 * st7 * ct * c7p) * V[n]; n += 1
        ylm += (0.72892666017f0 * st8 * c8p) * V[n]; n += 1

        # --- Ylm derivative with respect to theta (ylm_dt) ---
        n = 1
        ylm_dt += (0f0) * V[n]; n += 1

        ylm_dt += (0.54627421529f0 * 2f0 * stct * s2p) * V[n]; n += 1
        ylm_dt += (-1.0925484305920792f0 * (ct2 - st2) * sp) * V[n]; n += 1
        ylm_dt += (-0.31539156525252005f0 * 6f0 * ct * st) * V[n]; n += 1
        ylm_dt += (-1.0925484305920792f0 * (ct2 - st2) * cp) * V[n]; n += 1
        ylm_dt += (0.54627421529f0 * 2f0 * stct * c2p) * V[n]; n += 1

        ylm_dt += (0.62583573544f0 * 4f0 * st3 * ct * s4p) * V[n]; n += 1
        ylm_dt += (-1.77013076978f0 * (3f0 * st2 * ct2 - st4) * s3p) * V[n]; n += 1
        ylm_dt += (0.47308734787f0 * (2f0 * stct * (7f0 * ct2 - 1f0) - 14f0 * st3 * ct) * s2p) * V[n]; n += 1
        ylm_dt += (-0.66904654355f0 * ((ct2 - st2) * (7f0 * ct2 - 3f0) - (14f0 * st2 * ct2)) * sp) * V[n]; n += 1
        ylm_dt += (0.10578554691520431f0 * (-35f0 * 4f0 * ct2 * stct + 30f0 * 2f0 * stct)) * V[n]; n += 1
        ylm_dt += (-0.66904654355f0 * ((ct2 - st2) * (7f0 * ct2 - 3f0) - (14f0 * st2 * ct2)) * cp) * V[n]; n += 1
        ylm_dt += (0.47308734787f0 * (2f0 * stct * (7f0 * ct2 - 1f0) - 14f0 * st3 * ct) * c2p) * V[n]; n += 1
        ylm_dt += (-1.77013076978f0 * (3f0 * st2 * ct2 - st4) * c3p) * V[n]; n += 1
        ylm_dt += (0.62583573544f0 * 4f0 * st3 * ct * c4p) * V[n]; n += 1

        ylm_dt += (0.68318410519f0 * 6f0 * st5 * ct * s6p) * V[n]; n += 1
        ylm_dt += (-2.36661916223f0 * (5f0 * st4 * ct2 - st6) * s5p) * V[n]; n += 1
        ylm_dt += (0.50456490072f0 * (4f0 * st3 * ct * (11f0 * ct2 - 1f0) - 22f0 * st5 * ct) * s4p) * V[n]; n += 1
        ylm_dt += (-0.92120525951f0 * (3f0 * st2 * ct * (11f0 * ct3 - 3f0 * ct) - st4 * (33f0 * ct2 - 3f0)) * s3p) * V[n]; n += 1
        ylm_dt += (0.46060262975f0 * (2f0 * st * ct * (33f0 * ct4 - 18f0 * ct2 + 1f0) - st3 * (33f0 * 4f0 * ct3 - 36f0 * ct)) * s2p) * V[n]; n += 1
        ylm_dt += (-0.58262136251f0 * (ct * (33f0 * ct5 - 30f0 * ct3 + 5f0 * ct) - st2 * (33f0 * 5f0 * ct4 - 90f0 * ct2 + 5f0)) * sp) * V[n]; n += 1
        ylm_dt += (-0.06356920226f0 * (1386f0 * ct5 - 1260f0 * ct3 + 210f0 * ct) * st) * V[n]; n += 1
        ylm_dt += (-0.58262136251f0 * (ct * (33f0 * ct5 - 30f0 * ct3 + 5f0 * ct) - st2 * (33f0 * 5f0 * ct4 - 90f0 * ct2 + 5f0)) * cp) * V[n]; n += 1
        ylm_dt += (0.46060262975f0 * (2f0 * st * ct * (33f0 * ct4 - 18f0 * ct2 + 1f0) - st3 * (33f0 * 4f0 * ct3 - 36f0 * ct)) * c2p) * V[n]; n += 1
        ylm_dt += (-0.92120525951f0 * (3f0 * st2 * ct * (11f0 * ct3 - 3f0 * ct) - st4 * (33f0 * ct2 - 3f0)) * c3p) * V[n]; n += 1
        ylm_dt += (0.50456490072f0 * (4f0 * st3 * ct * (11f0 * ct2 - 1f0) - 22f0 * st5 * ct) * c4p) * V[n]; n += 1
        ylm_dt += (-2.36661916223f0 * (5f0 * st4 * ct2 - st6) * c5p) * V[n]; n += 1
        ylm_dt += (0.68318410519f0 * 6f0 * st5 * ct * c6p) * V[n]; n += 1

        ylm_dt += (0.72892666017f0 * 8f0 * st7 * ct * s8p) * V[n]; n += 1
        ylm_dt += (-2.9157066407f0 * (7f0 * st6 * ct2 - st8) * s7p) * V[n]; n += 1
        ylm_dt += (0.53233276606f0 * (6f0 * st5 * ct * (15f0 * ct2 - 1f0) - st7 * 30f0 * ct) * s6p) * V[n]; n += 1
        ylm_dt += (-3.4499106221f0 * (5f0 * st4 * ct * (5f0 * ct3 - ct) - st6 * (15f0 * ct2 - 1f0)) * s5p) * V[n]; n += 1
        ylm_dt += (0.47841652475f0 * (4f0 * st3 * ct * (65f0 * ct4 - 26f0 * ct2 + 1f0) - st5 * (260f0 * ct3 - 52f0 * ct)) * s4p) * V[n]; n += 1
        ylm_dt += (-1.2352661553f0 * (3f0 * st2 * ct * (39f0 * ct5 - 26f0 * ct3 + 3f0 * ct) - st4 * (195f0 * ct4 - 78f0 * ct2 + 3f0)) * s3p) * V[n]; n += 1
        ylm_dt += (0.45615225843f0 * (2f0 * st * ct * (143f0 * ct6 - 143f0 * ct4 + 33f0 * ct2 - 1f0) - st3 * (858f0 * ct5 - 572f0 * ct3 + 66f0 * ct)) * s2p) * V[n]; n += 1
        ylm_dt += (-0.10904124589f0 * (ct * (715f0 * ct7 - 1001f0 * ct5 + 385f0 * ct3 - 35f0 * ct) - st2 * (5005f0 * ct6 - 5005f0 * ct4 + 1155f0 * ct2 - 35f0)) * sp) * V[n]; n += 1
        ylm_dt += (-0.00908677049f0 * (8f0 * 6435f0 * ct7 - 6f0 * 12012f0 * ct5 + 4f0 * 6930f0 * ct3 - 2f0 * 1260f0 * ct) * st) * V[n]; n += 1
        ylm_dt += (-0.10904124589f0 * (ct * (715f0 * ct7 - 1001f0 * ct5 + 385f0 * ct3 - 35f0 * ct) - st2 * (5005f0 * ct6 - 5005f0 * ct4 + 1155f0 * ct2 - 35f0)) * cp) * V[n]; n += 1
        ylm_dt += (0.45615225843f0 * (2f0 * st * ct * (143f0 * ct6 - 143f0 * ct4 + 33f0 * ct2 - 1f0) - st3 * (858f0 * ct5 - 572f0 * ct3 + 66f0 * ct)) * c2p) * V[n]; n += 1
        ylm_dt += (-1.2352661553f0 * (3f0 * st2 * ct * (39f0 * ct5 - 26f0 * ct3 + 3f0 * ct) - st4 * (195f0 * ct4 - 78f0 * ct2 + 3f0)) * c3p) * V[n]; n += 1
        ylm_dt += (0.47841652475f0 * (4f0 * st3 * ct * (65f0 * ct4 - 26f0 * ct2 + 1f0) - st5 * (260f0 * ct3 - 52f0 * ct)) * c4p) * V[n]; n += 1
        ylm_dt += (-3.4499106221f0 * (5f0 * st4 * ct * (5f0 * ct3 - ct) - st6 * (15f0 * ct2 - 1f0)) * c5p) * V[n]; n += 1
        ylm_dt += (0.53233276606f0 * (6f0 * st5 * ct * (15f0 * ct2 - 1f0) - st7 * 30f0 * ct) * c6p) * V[n]; n += 1
        ylm_dt += (-2.9157066407f0 * (7f0 * st6 * ct2 - st8) * c7p) * V[n]; n += 1
        ylm_dt += (0.72892666017f0 * 8f0 * st7 * ct * c8p) * V[n]; n += 1
    end

    ylm_dp = _accum_ylm_dp(Val(DIV), V,
                           st, ct,
                           st2, st3, st4, st5, st6, st7, st8,
                           ct2, ct3, ct4, ct5, ct6, ct7, ct8,
                           stct,
                           sp, cp,
                           s2p, c2p,
                           s3p, c3p,
                           s4p, c4p,
                           s5p, c5p,
                           s6p, c6p,
                           s7p, c7p,
                           s8p, c8p)

    return ylm, ylm_dp, ylm_dt
end

function evaluate_for_diffusion(::SphericalHarmonics,
                                ::DirectFOD,
                                ϕ::𝒯,
                                θ::𝒯,
                                ::Int32,
                                voxels::Tuple{UInt32, UInt32, UInt32},
                                cache::ThreadedCache) where {𝒯}
    voxel_index₁, voxel_index₂, voxel_index₃ = voxels
    V = @view cache.odf[:, voxel_index₁, voxel_index₂, voxel_index₃]
    # we evaluate the derivative of the FODF. 
    # Careful that ∂F∂ϕ = ∂(FODF)∂ϕ / sin(θ) to avoid division by zero
    F, ∂F∂ϕ, ∂F∂θ = ishtmtx_dot_divst(ϕ, θ, V)

    # we apply a soft plus so that F > 0
    ∂ = ∂softplus(F, convert(𝒯, 100))
    F =  softplus(F, convert(𝒯, 100))
    ∂F∂θ *= ∂
    ∂F∂ϕ *= ∂

    ∫F  =  cache.odf[1, voxel_index₁, voxel_index₂, voxel_index₃]
    return F, ∂F∂ϕ, ∂F∂θ, ∫F
end

function evaluate_for_diffusion(::SphericalHarmonics,
                                ::PreComputeAllFOD,
                                ϕ::𝒯,
                                θ::𝒯,
                                ind_u::Int32,
                                voxels::Tuple{UInt32, UInt32, UInt32},
                                cache::ThreadedCache) where {𝒯}
    voxel_index₁, voxel_index₂, voxel_index₃ = voxels
    ∫F  =            cache.∫odf[voxel_index₁, voxel_index₂, voxel_index₃]
    F   =      cache.odf[ind_u, voxel_index₁, voxel_index₂, voxel_index₃]
    ∂F∂θ  =  cache.∂θodf[ind_u, voxel_index₁, voxel_index₂, voxel_index₃]
    ∂F∂ϕ =   cache.∂ϕodf[ind_u, voxel_index₁, voxel_index₂, voxel_index₃]
    # Careful that ∂F∂ϕ = ∂(FODF)∂ϕ / sin(θ) to avoid division by zero
    ∂F∂ϕ /= sin(θ)
    return F, ∂F∂ϕ, ∂F∂θ, ∫F
end