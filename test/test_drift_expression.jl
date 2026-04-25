# we solve ∂ₜ u = ∇ log f(u) for u ∈ S²
# for f(u) = 1 + ϵ u₃⋅u₁
using Test
using  Tractography, StaticArrays, LinearAlgebra
const TG = Tractography

function make_vector_field(shape=(30, 30, 3); n_coefficients = 45, Ty = Float64, ϵ = convert(Ty, 0.1))
    angles = TG.fibonacci_sampling(2000)
    bvectors = [SVector(TG.spherical_to_euclidean(d[1], d[2])...) for d in angles]
    fod_values = zeros(Ty, shape..., length(bvectors))

    fod_val_cst = [1 + ϵ * u[3]*u[1] for u in bvectors]
    for i in axes(fod_values,1), j in axes(fod_values,2), k in axes(fod_values, 3) 
        fod_values[i, j, k, :] = fod_val_cst
    end
    Yₗₘ = TG.get_vector_of_sh(angles, TG.get_lmax_from_fod_length(n_coefficients))
    fod_constant = Yₗₘ \ (@views fod_values[1, 1, 1, :])
    @assert fod_constant[1] ≈ (1+0/3) * sqrt(4pi)

    nx, ny, nz, nt = size(fod_values)
    fodv = reshape(fod_values, nx*ny*nz, nt)
    fod = zeros(Ty, nx, ny, nz, n_coefficients)
    for i in axes(fod, 1), j in axes(fod, 2)
        for k in axes(fod,3)
            @views fod[i,j,k,:] .= fod_constant
        end
    end
    mask = ones(Bool, size(fod)[1:3]...)
    return fod, mask
end

function euler_intrinsic(dt, t0, p0, ε, N)
    θ = t0
    φ = p0
    u₁, u₂, u₃ = TG.spherical_to_euclidean(t0, p0)
    U = SVector(u₁, u₂, u₃)
    sol = [copy(U)]
    for _ = 1:N-1
        F = 1 + ε * U[1] * U[3]
        # dU = SVector(U[3], 0, U[1]) - 2 * U[3]* U[1] * U
        dU = (I - U*U') * SVector(U[3], 0, U[1])
        dU = dU * ε / F
        U = @. U + dt * dU
        U = TG.normalize(U)
        θ, φ = TG.euclidean_to_spherical(U...)
        push!(sol, copy(U))
    end
    sol
end

𝒯 = Float64
_t0 = pi/3.3; _p0 = pi*0.89
_eps0 = 0.3
_fod, _mask = make_vector_field((2,2,2); ϵ = 𝒯(_eps0), Ty = 𝒯);

seeds = zeros(𝒯, 6, 3)
for i in axes(seeds, 2)
    seeds[:,i] .= vcat(10, 10, 10, TG.spherical_to_euclidean(_t0,_p0)...)
end

model_d = TG.Model(Δt = 𝒯(0.00025),
                foddata = TG.FODData(_fod, Array{𝒯}(1000*I(4)), zeros(4), false), # we put an enormous voxel size
                proba_min = 𝒯(0.0),
                evaluation_algo = TG.DirectFOD()
            )

nt = 60000
streamlines_transport, tract_length = @time TG.sample(model_d, TG.Transport(;γ = 𝒯(1) ), 𝒯.(seeds); nt, maxfod_start = false);

@test norm(diff(streamlines_transport[1, 1:end-1, 1])./model_d.Δt - 
        map(x->x[1], euler_intrinsic(model_d.Δt, _t0, _p0, _eps0, nt))[1:end-2], Inf) < model_d.Δt/2

@test norm(diff(streamlines_transport[2, 1:end-1, 1])./model_d.Δt - 
        map(x->x[2], euler_intrinsic(model_d.Δt, _t0, _p0, _eps0, nt))[1:end-2], Inf) < model_d.Δt/2

# case close to the north pole
_t0 = 0.0; _p0 = pi*0.89
_eps0 = 0.3
_fod, _mask = make_vector_field((2,2,2); ϵ = 𝒯(_eps0), Ty = 𝒯);

for i in axes(seeds, 2)
    seeds[:, i] .= vcat(10, 10, 10, TG.spherical_to_euclidean(_t0,_p0)...)
end

streamlines_transport, tract_length = @time TG.sample(model_d, TG.Transport(;γ = 𝒯(1) ), 𝒯.(seeds); nt, maxfod_start = false);

@test norm(diff(streamlines_transport[1, 1:end-1, 1])./model_d.Δt - 
        map(x->x[1], euler_intrinsic(model_d.Δt, _t0, _p0, _eps0, nt))[1:end-2], Inf) < model_d.Δt

@test norm(diff(streamlines_transport[2, 1:end-1, 1])./model_d.Δt - 
        map(x->x[2], euler_intrinsic(model_d.Δt, _t0, _p0, _eps0, nt))[1:end-2], Inf) < model_d.Δt

