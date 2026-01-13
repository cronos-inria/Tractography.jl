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

"""
$(TYPEDSIGNATURES)

Computes S = ∑Vₗₘ⋅Yₗₘ(θ, ϕ) and its partial derivatives ∂pS and ∂tS.

Returns (S, ∂pS, ∂tS)
"""
function ishtmtx_dot(phi::𝒯, 
                     theta::𝒯,
                     V::AbstractVector{𝒯},
                     ) where {𝒯}

    st, ct = sincos(theta)
    sp, cp = sincos(phi)

    if false
        # sin(m*phi), cos(m*phi)
        s2p, c2p = sincos(2 * phi)
        s3p, c3p = sincos(3 * phi)
        s4p, c4p = sincos(4 * phi)
        s5p, c5p = sincos(5 * phi)
        s6p, c6p = sincos(6 * phi)
        s7p, c7p = sincos(7 * phi)
        s8p, c8p = sincos(8 * phi)
    else

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
    end


    # Precompute powers
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

    ylm = ylm_dp = ylm_dt = zero(𝒯)

    @inbounds begin
        # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
        # https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
        # what is the normalization?

        # --- Ylm values (indices +1 vs C) ---
        n = 1
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

        # --- Ylm derivative with respect to phi (ylm_dp) ---
        n = 1
        ylm_dp += (0f0) * V[n]; n += 1

        ylm_dp += (0.54627421529f0 * st2 * 2f0 * c2p) * V[n]; n += 1
        ylm_dp += (-1.0925484305920792f0 * stct * cp) * V[n]; n += 1
        ylm_dp += (0f0) * V[n]; n += 1
        ylm_dp += (1.0925484305920792f0 * stct * sp) * V[n]; n += 1
        ylm_dp += (-0.54627421529f0 * st2 * 2f0 * s2p) * V[n]; n += 1

        ylm_dp += (0.62583573544f0 * st4 * 4f0 * c4p) * V[n]; n += 1
        ylm_dp += (-1.77013076978f0 * st3 * ct * 3f0 * c3p) * V[n]; n += 1
        ylm_dp += (0.47308734787f0 * st2 * (7f0 * ct2 - 1f0) * 2f0 * c2p) * V[n]; n += 1
        ylm_dp += (-0.66904654355f0 * stct * (7f0 * ct2 - 3f0) * cp) * V[n]; n += 1
        ylm_dp += (0) * V[n]; n += 1
        ylm_dp += (0.66904654355f0 * stct * (7f0 * ct2 - 3f0) * sp) * V[n]; n += 1
        ylm_dp += (-0.47308734787f0 * st2 * (7f0 * ct2 - 1f0) * 2f0 * s2p) * V[n]; n += 1
        ylm_dp += (1.77013076978f0 * st3 * ct * 3f0 * s3p) * V[n]; n += 1
        ylm_dp += (-0.62583573544f0 * st4 * 4f0 * s4p) * V[n]; n += 1

        ylm_dp += (0.68318410519f0 * st6 * 6f0 * c6p) * V[n]; n += 1
        ylm_dp += (-2.36661916223f0 * st5 * ct * 5f0 * c5p) * V[n]; n += 1
        ylm_dp += (0.50456490072f0 * st4 * (11f0 * ct2 - 1f0) * 4f0 * c4p) * V[n]; n += 1
        ylm_dp += (-0.92120525951f0 * st3 * (11f0 * ct3 - 3f0 * ct) * 3f0 * c3p) * V[n]; n += 1
        ylm_dp += (0.46060262975f0 * st2 * (33f0 * ct4 - 18f0 * ct2 + 1f0) * 2f0 * c2p) * V[n]; n += 1
        ylm_dp += (-0.58262136251f0 * st * (33f0 * ct5 - 30f0 * ct3 + 5f0 * ct) * cp) * V[n]; n += 1
        ylm_dp += (0f0) * V[n]; n += 1
        ylm_dp += (0.58262136251f0 * st * (33f0 * ct5 - 30f0 * ct3 + 5f0 * ct) * sp) * V[n]; n += 1
        ylm_dp += (-0.46060262975f0 * st2 * (33f0 * ct4 - 18f0 * ct2 + 1f0) * 2f0 * s2p) * V[n]; n += 1
        ylm_dp += (0.92120525951f0 * st3 * (11f0 * ct3 - 3f0 * ct) * 3f0 * s3p) * V[n]; n += 1
        ylm_dp += (-0.50456490072f0 * st4 * (11f0 * ct2 - 1f0) * 4f0 * s4p) * V[n]; n += 1
        ylm_dp += (2.36661916223f0 * st5 * ct * 5f0 * s5p) * V[n]; n += 1
        ylm_dp += (-0.68318410519f0 * st6 * 6f0 * s6p) * V[n]; n += 1

        ylm_dp += (0.72892666017f0 * st8 * 8f0 * c8p) * V[n]; n += 1
        ylm_dp += (-2.9157066407f0 * st7 * ct * 7f0 * c7p) * V[n]; n += 1
        ylm_dp += (0.53233276606f0 * st6 * (15f0 * ct2 - 1f0) * 6f0 * c6p) * V[n]; n += 1
        ylm_dp += (-3.4499106221f0 * st5 * (5f0 * ct3 - ct) * 5f0 * c5p) * V[n]; n += 1
        ylm_dp += (0.47841652475f0 * st4 * (65f0 * ct4 - 26f0 * ct2 + 1f0) * 4f0 * c4p) * V[n]; n += 1
        ylm_dp += (-1.2352661553f0 * st3 * (39f0 * ct5 - 26f0 * ct3 + 3f0 * ct) * 3f0 * c3p) * V[n]; n += 1
        ylm_dp += (0.45615225843f0 * st2 * (143f0 * ct6 - 143f0 * ct4 + 33f0 * ct2 - 1f0) * 2f0 * c2p) * V[n]; n += 1
        ylm_dp += (-0.10904124589f0 * st * (715f0 * ct7 - 1001f0 * ct5 + 385f0 * ct3 - 35f0 * ct) * cp) * V[n]; n += 1
        ylm_dp += (0f0) * V[n]; n += 1
        ylm_dp += (0.10904124589f0 * st * (715f0 * ct7 - 1001f0 * ct5 + 385f0 * ct3 - 35f0 * ct) * sp) * V[n]; n += 1
        ylm_dp += (-0.45615225843f0 * st2 * (143f0 * ct6 - 143f0 * ct4 + 33f0 * ct2 - 1f0) * 2f0 * s2p) * V[n]; n += 1
        ylm_dp += (1.2352661553f0 * st3 * (39f0 * ct5 - 26f0 * ct3 + 3f0 * ct) * 3f0 * s3p) * V[n]; n += 1
        ylm_dp += (-0.47841652475f0 * st4 * (65f0 * ct4 - 26f0 * ct2 + 1f0) * 4f0 * s4p) * V[n]; n += 1
        ylm_dp += (3.4499106221f0 * st5 * (5f0 * ct3 - ct) * 5f0 * s5p) * V[n]; n += 1
        ylm_dp += (-0.53233276606f0 * st6 * (15f0 * ct2 - 1f0) * 6f0 * s6p) * V[n]; n += 1
        ylm_dp += (2.9157066407f0 * st7 * ct * 7f0 * s7p) * V[n]; n += 1
        ylm_dp += (-0.72892666017f0 * st8 * 8f0 * s8p) * V[n]; n += 1

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

    return ylm, ylm_dp, ylm_dt
end