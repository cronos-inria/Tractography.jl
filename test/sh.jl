using Tractography, Test, LinearAlgebra
const TG = Tractography


TG.∂θro_sh(0.,0.,1,1)
TG.∂ϕro_sh(0.,0.,1,1)
TG.get_vector_of_sh([(0.,0.), (0.1, 0.1)], 8, 0)
TG.get_vector_of_sh([(0.,0.), (0.1, 0.1)], 8, 1)
TG.get_vector_of_sh([(0.,0.), (0.1, 0.1)], 8, 2)

# test of the value of real spherical harmonics
# https://userdocs.mrtrix.org/en/latest/concepts/spherical_harmonics.html
# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
# https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
# careful, $Y_l^m \neq Y_{l,m}$
my_ylm(t,p,ii) = TG.get_vector_of_sh([(t,p)],8)[ii]

θ = 0.1f0; ϕ = 0.2f0
@test my_ylm(θ, ϕ, 1) ≈ sqrt(1/4/pi)

# recall that real spherical harmonics Y2,-2 ≡ sqrt(15/pi)/4*sin(2ϕ)*sin(θ)^2
fact_sh(l,m) = sqrt((2l+1)//4/pi * factorial(l-m)//factorial(l+m) )
@test fact_sh(2,-2) * 3/24 * sqrt(2) ≈ sqrt(15/pi)/4

@test my_ylm(θ, ϕ, 2) ≈ sqrt(15/pi)/4*sin(2ϕ)*sin(θ)^2
@test my_ylm(θ, ϕ, 2) ≈ 0.54627424*  sin(2ϕ)*(sin(θ)^2)

@test my_ylm(θ, ϕ, 3) ≈ -0.54627424*sin(2θ)*sin(ϕ)
@test my_ylm(θ, ϕ, 3) ≈ -sqrt(15/pi)/4*sin(2θ)*sin(ϕ)

@test my_ylm(θ, ϕ, 4) ≈ -0.31539157 + 0.94617474(cos(θ)^2)

@test my_ylm(θ, ϕ, 5) ≈ -sqrt(15/pi)/4*cos(ϕ)*sin(2θ)
@test my_ylm(θ, ϕ, 6) ≈ 0.54627424(sin(θ)^2)*cos(2ϕ)

@test my_ylm(θ, ϕ, 7) ≈ 0.6258357sin(4ϕ)*(sin(θ)^4)
@test my_ylm(θ, ϕ, 8) ≈ -1.7701306cos(θ)*sin(3ϕ)*(sin(θ)^3)
@test my_ylm(θ, ϕ, 9) ≈ -0.47308737sin(2ϕ)*(sin(θ)^2) + 3.3116114(cos(θ)^2)*sin(2ϕ)*(sin(θ)^2)
@test my_ylm(θ, ϕ, 10) ≈ 1.0035698sin(2θ)*sin(ϕ) - 4.683326(cos(θ)^3)*sin(θ)*sin(ϕ)
@test my_ylm(θ, ϕ, 11) ≈ 0.31735665 - 3.1735663(cos(θ)^2) + 3.7024941(cos(θ)^4)
@test my_ylm(θ, ϕ, 12) ≈ 1.0035698cos(ϕ)*sin(2θ) - 4.683326(cos(θ)^3)*cos(ϕ)*sin(θ)
@test my_ylm(θ, ϕ, 13) ≈ -0.47308737(sin(θ)^2)*cos(2ϕ) + 3.3116114(cos(θ)^2)*(sin(θ)^2)*cos(2ϕ)
@test my_ylm(θ, ϕ, 14) ≈ -1.7701306cos(θ)*(sin(θ)^3)*cos(3ϕ)
@test my_ylm(θ, ϕ, 15) ≈ 0.6258357(sin(θ)^4)*cos(4ϕ)
#######################################################
function ishtmtx(phi::𝒯, 
                 theta::𝒯,
                 ylm::Vector{𝒯},
                 ylm_dp::Vector{𝒯},
                 ylm_dt::Vector{𝒯}) where {𝒯}
    @assert length(ylm) == length(ylm_dp) == length(ylm_dt) == 45

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

    @inbounds begin

        # https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
        # c'est quoi la normalization??

        # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics


        # --- Ylm values (indices +1 vs C) ---
        ylm[1]  = 𝒯(1/sqrt(4pi))

        # Real spherical harmonics
        # 1/4*sqrt(15/pi) = 0.5462742152960396
        ylm[2]  = 0.54627421529f0 * st2 * s2p
        # 1/2*sqrt(15/2/pi) = 0.7725484040463791
        ylm[3]  = -1.0925484305920792f0 * stct * sp
        ylm[4]  = 0.31539156525252005f0 * (3 * ct2 - 1)
        ylm[5]  = -1.0925484305920792f0 * stct * cp #sqrt(15/pi)/2
        ylm[6]  = 0.54627421529f0 * st2 * c2p

        ylm[7]  = 0.62583573544f0  * st4 * s4p
        ylm[8]  = -1.77013076978f0 * st3 * ct * s3p
        ylm[9]  = 0.47308734787f0  * st2 * (7 * ct2 - 1) * s2p
        ylm[10] = -0.66904654355f0 * st  * (7 * ct2 - 3) * ct * sp
        ylm[11] = 0.10578554691520431f0  * (35 * ct2 * ct2 - 30 * ct2 + 3)
        ylm[12] = -0.66904654355f0 * st  * (7 * ct2 - 3) * ct * cp
        ylm[13] = 0.47308734787f0  * st2 * (7 * ct2 - 1) * c2p
        ylm[14] = -1.77013076978f0 * st3 * ct * c3p
        ylm[15] = 0.62583573544f0  * st4 * c4p

        ylm[16] = 0.68318410519f0  * st6 * s6p
        ylm[17] = -2.36661916223f0 * st5 * ct * s5p
        ylm[18] = 0.50456490072f0  * st4 * (11 * ct2 - 1) * s4p
        ylm[19] = -0.92120525951f0 * st3 * (11 * ct3 - 3 * ct) * s3p
        ylm[20] = 0.46060262975f0  * st2 * (33 * ct4 - 18 * ct2 + 1) * s2p
        ylm[21] = -0.58262136251f0 * st * (33 * ct5 - 30 * ct3 + 5 * ct) * sp
        ylm[22] = 0.06356920226f0  * (231 * ct6 - 315 * ct4 + 105 * ct2 - 5)
        ylm[23] = -0.58262136251f0 * st * (33 * ct5 - 30 * ct3 + 5 * ct) * cp
        ylm[24] = 0.46060262975f0  * st2 * (33 * ct4 - 18 * ct2 + 1) * c2p
        ylm[25] = -0.92120525951f0 * st3 * (11 * ct3 - 3 * ct) * c3p
        ylm[26] = 0.50456490072f0  * st4 * (11 * ct2 - 1) * c4p
        ylm[27] = -2.36661916223f0 * st5 * ct * c5p
        ylm[28] = 0.68318410519f0  * st6 * c6p

        ylm[29] = 0.72892666017f0 * st8 * s8p
        ylm[30] = -2.9157066407f0 * st7 * ct * s7p
        ylm[31] = 0.53233276606f0 * st6 * (15f0 * ct2 - 1f0) * s6p
        ylm[32] = -3.4499106221f0 * st5 * (5f0 * ct3 - ct) * s5p
        ylm[33] = 0.47841652475f0 * st4 * (65f0 * ct4 - 26f0 * ct2 + 1f0) * s4p
        ylm[34] = -1.2352661553f0 * st3 * (39f0 * ct5 - 26f0 * ct3 + 3f0 * ct) * s3p
        ylm[35] = 0.45615225843f0 * st2 * (143f0 * ct6 - 143f0 * ct4 + 33f0 * ct2 - 1f0) * s2p
        ylm[36] = -0.10904124589f0 * st * (715f0 * ct7 - 1001f0 * ct5 + 385f0 * ct3 - 35f0 * ct) * sp
        ylm[37] = 0.00908677049f0 * (6435f0 * ct8 - 12012f0 * ct6 + 6930f0 * ct4 - 1260f0 * ct2 + 35f0)
        ylm[38] = -0.10904124589f0 * st * (715f0 * ct7 - 1001f0 * ct5 + 385f0 * ct3 - 35f0 * ct) * cp
        ylm[39] = 0.45615225843f0 * st2 * (143f0 * ct6 - 143f0 * ct4 + 33f0 * ct2 - 1f0) * c2p
        ylm[40] = -1.2352661553f0 * st3 * (39f0 * ct5 - 26f0 * ct3 + 3f0 * ct) * c3p
        ylm[41] = 0.47841652475f0 * st4 * (65f0 * ct4 - 26f0 * ct2 + 1f0) * c4p
        ylm[42] = -3.4499106221f0 * st5 * (5f0 * ct3 - ct) * c5p
        ylm[43] = 0.53233276606f0 * st6 * (15f0 * ct2 - 1f0) * c6p
        ylm[44] = -2.9157066407f0 * st7 * ct * c7p
        ylm[45] = 0.72892666017f0 * st8 * c8p

        # --- Ylm derivative with respect to phi (ylm_dp) ---
        ylm_dp[1]  = 0f0

        ylm_dp[2]  = 0.54627421529f0 * st2 * 2f0 * c2p
        ylm_dp[3]  = -1.0925484305920792f0 * stct * cp
        ylm_dp[4]  = 0f0
        ylm_dp[5]  = 1.0925484305920792f0 * stct * sp
        ylm_dp[6]  = -0.54627421529f0 * st2 * 2f0 * s2p

        ylm_dp[7]  = 0.62583573544f0 * st4 * 4f0 * c4p
        ylm_dp[8]  = -1.77013076978f0 * st3 * ct * 3f0 * c3p
        ylm_dp[9]  = 0.47308734787f0 * st2 * (7f0 * ct2 - 1f0) * 2f0 * c2p
        ylm_dp[10] = -0.66904654355f0 * stct * (7f0 * ct2 - 3f0) * cp
        ylm_dp[11] = 0f0
        ylm_dp[12] = 0.66904654355f0 * stct * (7f0 * ct2 - 3f0) * sp
        ylm_dp[13] = -0.47308734787f0 * st2 * (7f0 * ct2 - 1f0) * 2f0 * s2p
        ylm_dp[14] = 1.77013076978f0 * st3 * ct * 3f0 * s3p
        ylm_dp[15] = -0.62583573544f0 * st4 * 4f0 * s4p

        ylm_dp[16] = 0.68318410519f0 * st6 * 6f0 * c6p
        ylm_dp[17] = -2.36661916223f0 * st5 * ct * 5f0 * c5p
        ylm_dp[18] = 0.50456490072f0 * st4 * (11f0 * ct2 - 1f0) * 4f0 * c4p
        ylm_dp[19] = -0.92120525951f0 * st3 * (11f0 * ct3 - 3f0 * ct) * 3f0 * c3p
        ylm_dp[20] = 0.46060262975f0 * st2 * (33f0 * ct4 - 18f0 * ct2 + 1f0) * 2f0 * c2p
        ylm_dp[21] = -0.58262136251f0 * st * (33f0 * ct5 - 30f0 * ct3 + 5f0 * ct) * cp
        ylm_dp[22] = 0f0
        ylm_dp[23] = 0.58262136251f0 * st * (33f0 * ct5 - 30f0 * ct3 + 5f0 * ct) * sp
        ylm_dp[24] = -0.46060262975f0 * st2 * (33f0 * ct4 - 18f0 * ct2 + 1f0) * 2f0 * s2p
        ylm_dp[25] = 0.92120525951f0 * st3 * (11f0 * ct3 - 3f0 * ct) * 3f0 * s3p
        ylm_dp[26] = -0.50456490072f0 * st4 * (11f0 * ct2 - 1f0) * 4f0 * s4p
        ylm_dp[27] = 2.36661916223f0 * st5 * ct * 5f0 * s5p
        ylm_dp[28] = -0.68318410519f0 * st6 * 6f0 * s6p

        ylm_dp[29] = 0.72892666017f0 * st8 * 8f0 * c8p
        ylm_dp[30] = -2.9157066407f0 * st7 * ct * 7f0 * c7p
        ylm_dp[31] = 0.53233276606f0 * st6 * (15f0 * ct2 - 1f0) * 6f0 * c6p
        ylm_dp[32] = -3.4499106221f0 * st5 * (5f0 * ct3 - ct) * 5f0 * c5p
        ylm_dp[33] = 0.47841652475f0 * st4 * (65f0 * ct4 - 26f0 * ct2 + 1f0) * 4f0 * c4p
        ylm_dp[34] = -1.2352661553f0 * st3 * (39f0 * ct5 - 26f0 * ct3 + 3f0 * ct) * 3f0 * c3p
        ylm_dp[35] = 0.45615225843f0 * st2 * (143f0 * ct6 - 143f0 * ct4 + 33f0 * ct2 - 1f0) * 2f0 * c2p
        ylm_dp[36] = -0.10904124589f0 * st * (715f0 * ct7 - 1001f0 * ct5 + 385f0 * ct3 - 35f0 * ct) * cp
        ylm_dp[37] = 0f0
        ylm_dp[38] = 0.10904124589f0 * st * (715f0 * ct7 - 1001f0 * ct5 + 385f0 * ct3 - 35f0 * ct) * sp
        ylm_dp[39] = -0.45615225843f0 * st2 * (143f0 * ct6 - 143f0 * ct4 + 33f0 * ct2 - 1f0) * 2f0 * s2p
        ylm_dp[40] = 1.2352661553f0 * st3 * (39f0 * ct5 - 26f0 * ct3 + 3f0 * ct) * 3f0 * s3p
        ylm_dp[41] = -0.47841652475f0 * st4 * (65f0 * ct4 - 26f0 * ct2 + 1f0) * 4f0 * s4p
        ylm_dp[42] = 3.4499106221f0 * st5 * (5f0 * ct3 - ct) * 5f0 * s5p
        ylm_dp[43] = -0.53233276606f0 * st6 * (15f0 * ct2 - 1f0) * 6f0 * s6p
        ylm_dp[44] = 2.9157066407f0 * st7 * ct * 7f0 * s7p
        ylm_dp[45] = -0.72892666017f0 * st8 * 8f0 * s8p

        # --- Ylm derivative with respect to theta (ylm_dt) ---
        ylm_dt[1]  = 0f0

        ylm_dt[2]  = 0.54627421529f0 * 2f0 * stct * s2p
        ylm_dt[3]  = -1.0925484305920792f0 * (ct2 - st2) * sp
        ylm_dt[4]  = -0.31539156525252005f0 * 6f0 * ct * st
        ylm_dt[5]  = -1.0925484305920792f0 * (ct2 - st2) * cp
        ylm_dt[6]  = 0.54627421529f0 * 2f0 * stct * c2p

        ylm_dt[7]  = 0.62583573544f0 * 4f0 * st3 * ct * s4p
        ylm_dt[8]  = -1.77013076978f0 * (3f0 * st2 * ct2 - st4) * s3p
        ylm_dt[9]  = 0.47308734787f0 * (2f0 * stct * (7f0 * ct2 - 1f0) - 14f0 * st3 * ct) * s2p
        ylm_dt[10] = -0.66904654355f0 * ((ct2 - st2) * (7f0 * ct2 - 3f0) - (14f0 * st2 * ct2)) * sp
        ylm_dt[11] = 0.10578554691520431f0 * (-35f0 * 4f0 * ct2 * stct + 30f0 * 2f0 * stct)
        ylm_dt[12] = -0.66904654355f0 * ((ct2 - st2) * (7f0 * ct2 - 3f0) - (14f0 * st2 * ct2)) * cp
        ylm_dt[13] = 0.47308734787f0 * (2f0 * stct * (7f0 * ct2 - 1f0) - 14f0 * st3 * ct) * c2p
        ylm_dt[14] = -1.77013076978f0 * (3f0 * st2 * ct2 - st4) * c3p
        ylm_dt[15] = 0.62583573544f0 * 4f0 * st3 * ct * c4p

        ylm_dt[16] = 0.68318410519f0 * 6f0 * st5 * ct * s6p
        ylm_dt[17] = -2.36661916223f0 * (5f0 * st4 * ct2 - st6) * s5p
        ylm_dt[18] = 0.50456490072f0 * (4f0 * st3 * ct * (11f0 * ct2 - 1f0) - 22f0 * st5 * ct) * s4p
        ylm_dt[19] = -0.92120525951f0 * (3f0 * st2 * ct * (11f0 * ct3 - 3f0 * ct) - st4 * (33f0 * ct2 - 3f0)) * s3p
        ylm_dt[20] = 0.46060262975f0 * (2f0 * st * ct * (33f0 * ct4 - 18f0 * ct2 + 1f0) - st3 * (33f0 * 4f0 * ct3 - 36f0 * ct)) * s2p
        ylm_dt[21] = -0.58262136251f0 * (ct * (33f0 * ct5 - 30f0 * ct3 + 5f0 * ct) - st2 * (33f0 * 5f0 * ct4 - 90f0 * ct2 + 5f0)) * sp
        ylm_dt[22] = -0.06356920226f0 * (1386f0 * ct5 - 1260f0 * ct3 + 210f0 * ct) * st
        ylm_dt[23] = -0.58262136251f0 * (ct * (33f0 * ct5 - 30f0 * ct3 + 5f0 * ct) - st2 * (33f0 * 5f0 * ct4 - 90f0 * ct2 + 5f0)) * cp
        ylm_dt[24] = 0.46060262975f0 * (2f0 * st * ct * (33f0 * ct4 - 18f0 * ct2 + 1f0) - st3 * (33f0 * 4f0 * ct3 - 36f0 * ct)) * c2p
        ylm_dt[25] = -0.92120525951f0 * (3f0 * st2 * ct * (11f0 * ct3 - 3f0 * ct) - st4 * (33f0 * ct2 - 3f0)) * c3p
        ylm_dt[26] = 0.50456490072f0 * (4f0 * st3 * ct * (11f0 * ct2 - 1f0) - 22f0 * st5 * ct) * c4p
        ylm_dt[27] = -2.36661916223f0 * (5f0 * st4 * ct2 - st6) * c5p
        ylm_dt[28] = 0.68318410519f0 * 6f0 * st5 * ct * c6p

        ylm_dt[29] = 0.72892666017f0 * 8f0 * st7 * ct * s8p
        ylm_dt[30] = -2.9157066407f0 * (7f0 * st6 * ct2 - st8) * s7p
        ylm_dt[31] = 0.53233276606f0 * (6f0 * st5 * ct * (15f0 * ct2 - 1f0) - st7 * 30f0 * ct) * s6p
        ylm_dt[32] = -3.4499106221f0 * (5f0 * st4 * ct * (5f0 * ct3 - ct) - st6 * (15f0 * ct2 - 1f0)) * s5p
        ylm_dt[33] = 0.47841652475f0 * (4f0 * st3 * ct * (65f0 * ct4 - 26f0 * ct2 + 1f0) - st5 * (260f0 * ct3 - 52f0 * ct)) * s4p
        ylm_dt[34] = -1.2352661553f0 * (3f0 * st2 * ct * (39f0 * ct5 - 26f0 * ct3 + 3f0 * ct) - st4 * (195f0 * ct4 - 78f0 * ct2 + 3f0)) * s3p
        ylm_dt[35] = 0.45615225843f0 * (2f0 * st * ct * (143f0 * ct6 - 143f0 * ct4 + 33f0 * ct2 - 1f0) - st3 * (858f0 * ct5 - 572f0 * ct3 + 66f0 * ct)) * s2p
        ylm_dt[36] = -0.10904124589f0 * (ct * (715f0 * ct7 - 1001f0 * ct5 + 385f0 * ct3 - 35f0 * ct) - st2 * (5005f0 * ct6 - 5005f0 * ct4 + 1155f0 * ct2 - 35f0)) * sp
        ylm_dt[37] = -0.00908677049f0 * (8f0 * 6435f0 * ct7 - 6f0 * 12012f0 * ct5 + 4f0 * 6930f0 * ct3 - 2f0 * 1260f0 * ct) * st
        ylm_dt[38] = -0.10904124589f0 * (ct * (715f0 * ct7 - 1001f0 * ct5 + 385f0 * ct3 - 35f0 * ct) - st2 * (5005f0 * ct6 - 5005f0 * ct4 + 1155f0 * ct2 - 35f0)) * cp
        ylm_dt[39] = 0.45615225843f0 * (2f0 * st * ct * (143f0 * ct6 - 143f0 * ct4 + 33f0 * ct2 - 1f0) - st3 * (858f0 * ct5 - 572f0 * ct3 + 66f0 * ct)) * c2p
        ylm_dt[40] = -1.2352661553f0 * (3f0 * st2 * ct * (39f0 * ct5 - 26f0 * ct3 + 3f0 * ct) - st4 * (195f0 * ct4 - 78f0 * ct2 + 3f0)) * c3p
        ylm_dt[41] = 0.47841652475f0 * (4f0 * st3 * ct * (65f0 * ct4 - 26f0 * ct2 + 1f0) - st5 * (260f0 * ct3 - 52f0 * ct)) * c4p
        ylm_dt[42] = -3.4499106221f0 * (5f0 * st4 * ct * (5f0 * ct3 - ct) - st6 * (15f0 * ct2 - 1f0)) * c5p
        ylm_dt[43] = 0.53233276606f0 * (6f0 * st5 * ct * (15f0 * ct2 - 1f0) - st7 * 30f0 * ct) * c6p
        ylm_dt[44] = -2.9157066407f0 * (7f0 * st6 * ct2 - st8) * c7p
        ylm_dt[45] = 0.72892666017f0 * 8f0 * st7 * ct * c8p
    end

    return nothing
end

let
    _Y1 = zeros(Float32, 45); _Y2 = zero(_Y1); _Y3 = zero(_Y1); _V = rand(Float32, 45)
    _theta = rand(Float32); _phi = rand(Float32); _Y1 .= 0
    ishtmtx(_phi, _theta, _Y1, _Y2, _Y3)
    r1 = TG.ishtmtx_dot(_phi, _theta, _V)
    @test (@allocated TG.ishtmtx_dot(_phi, _theta, _V)) == 0
    @test dot(_Y1, _V) ≈ r1[1]
    @test dot(_Y2, _V) ≈ r1[2]
    @test dot(_Y3, _V) ≈ r1[3]

    @test (@allocated TG.ishtmtx_dot_divst(_phi, _theta, _V)) == 0
    r1 = TG.ishtmtx_dot_divst(_phi, _theta, _V)
    @test dot(_Y1, _V) ≈ r1[1]
    @test dot(_Y2, _V) ≈ r1[2] * sin(_theta)
    @test dot(_Y3, _V) ≈ r1[3]
end