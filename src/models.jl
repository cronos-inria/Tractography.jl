abstract type AbstractCache end
# evaluation of spherical harmonics
abstract type AbstractFODEvaluation end

"""
$(TYPEDEF)

The evaluation of the basis (for example spherical harmonics) is done on the fly. Requires little memory.

See also `PreComputeAllFOD`
"""
struct DirectFOD <: AbstractFODEvaluation end

"""
$(TYPEDEF)

Set up for plotting ODF.
"""
struct PlottingFOD <: AbstractFODEvaluation end

"""
$(TYPEDEF)

Evaluation of the basis based on Fibonacci sampling. All ODF are pre-computed once and saved in a cache. Their positivity is enforced with a `max(0,⋅)` or a mollifier. 

!!! danger 
    Requires a relatively large memory!

## Details
If you have `na` angles for sampling the unit sphere and the data is of size `(nx, ny, nz, nsph)`, it yields a matrix of dimensions `(nx, ny, nz, na)`.
"""
struct PreComputeAllFOD <: AbstractFODEvaluation end
####################################################################################################
# streamlines tracking algorithms
abstract type AbstractSampler end
# sampler that are based on a grid. Basically everything except ::Rejection
abstract type AbstractNotPureRejectionSampler <: AbstractSampler end
# Deterministic samplers
abstract type DeterministicSampler <: AbstractNotPureRejectionSampler end
# sampling based on SDE
abstract type AbstractSDESampler{T} <: AbstractSampler end
abstract type AbstractSDESamplerOrder2{T} <: AbstractSDESampler{T} end

"""
$(TYPEDEF)

Tractography based sampling of structural connectivity. 
Do not compute the full streamline but only return the first/last points and the streamlines lengths. This allows to compute many more streamlines on GPU where memory is limited.

## Constructor example
 - `Connectivity(Probabilistic())`
"""
struct Connectivity{Talg} <: AbstractSampler
    alg::Talg
end

_get_alg(alg) = alg
_get_alg(alg::Connectivity) = alg.alg

"""
$(TYPEDEF)

Tractography sampling of the Tractography Markov Chain (TMC) performed with the cumulative sum distribution. Can be used with the basis evalution strategy `PreComputeAllFOD`.

# Constructor

`Probabilistic()`
"""
struct Probabilistic <: AbstractNotPureRejectionSampler end

"""
$(TYPEDEF)

Tractography sampling of the Tractography Markov Chain (TMC) performed with the argmax function. Can be used with the basis evalution strategy `PreComputeAllFOD`.
"""
struct Deterministic <: DeterministicSampler end

"""
$(TYPEDEF)

Tractography sampling of the diffusive model performed with geometric Euler-Maruyama method [1]; its precision is weak order 1. The streamlines (Xₜ)ₜ are solution of the SDE

dXₜ = Uₜ⋅dt

dUₜ = γ⋅∇log f(Uₜ)⋅dt + √(2γ ⋅ γ_noise) ⋅ dnoiseₜ

# Arguments (with default values):
$(TYPEDFIELDS)

# Constructor

Example for `Float32`: `Diffusion(γ = 1f0)`.
If you want `Float64`, you have to pass the two scalars

    ```Diffusion(γ = 1.0, γ_noise = 1.0)```

# Reference(s)
[1] Bharath, K., Lewis, A., Sharma, A., & Tretyakov, M. V. (n.d.). Sampling and Estimation on Manifolds using the Langevin Diﬀusion.
"""
@with_kw_noshow struct Diffusion{T, Tmol, Tdmol} <: AbstractSDESampler{T}
    "γ parameter of the diffusion process. It is related to the curvature of the streamline."
    γ::T = 1f0
    "parameter of the diffusion process to scale the variance."
    γ_noise::T = 1f0
    "mollifier."
    mollifier::Tmol = Base.Fix2(softplus, 10)
    "differential of mollifier."
    d_mollifier::Tdmol = Base.Fix2(∂softplus, 10)
    "Fixed time step?"
    adaptive::Bool = false
end

get_γ(alg::AbstractSDESampler) = alg.γ
get_γ(alg::Connectivity) = get_γ(_get_alg(alg))
get_γ_noise(alg::AbstractSDESampler) = alg.γ_noise
get_γ_noise(alg::Connectivity) = get_γ_noise(_get_alg(alg))
is_adaptive(alg::AbstractSDESampler) = alg.adaptive
is_adaptive(alg::Connectivity) = is_adaptive(_get_alg(alg))

"""
$(TYPEDSIGNATURES)

Define the transport algorithm. Options are the same as for `Diffusion`.

# Arguments (with default values):
$(TYPEDFIELDS)
"""
@with_kw_noshow struct Transport{T, Tmol, Tdmol} <: AbstractSDESampler{T}
    "γ parameter of the diffusion process. It is related to the curvature of the streamline."
    γ::T = 1f0
    "mollifier."
    mollifier::Tmol = Base.Fix2(softplus, 10)
    "differential of mollifier."
    d_mollifier::Tdmol = Base.Fix2(∂softplus, 10)
    "Fixed time step?"
    adaptive::Bool = false
end
get_γ_noise(alg::Transport{T}) where {T} = zero(T)

function Base.show(io::IO, alg::AbstractSDESampler{T}) where {T}
    printstyled(io, "$(typeof(alg).name.name) [$T]" ; bold = true, color = :cyan)
    printstyled(io, " sampling algorithm\n", color = :cyan)
    println(io, "├─ adaptive = ", is_adaptive(alg))
    if ~(alg isa Transport)
        println(io, "├─ γ        = ", get_γ(alg))
        println(io, "└─ γ_noise  = ", get_γ_noise(alg))
    else
        println(io, "└─ γ        = ", get_γ(alg))
    end
end

"""
$(TYPEDEF)

Tractography sampling of the diffusive model performed with Frozen-Flow method method SFF2 [1]; its precision is weak order 2. The streamlines (Xₜ)ₜ are solution of the SDE

dXₜ = Uₜ⋅dt

dUₜ = γ⋅∇log f(Uₜ)⋅dt + √(2γ ⋅ γ_noise) ⋅ dnoiseₜ

# Arguments (with default values):
$(TYPEDFIELDS)

# Constructor

Example for `Float32`: `SFF2(γ = 1f0)`.
If you want `Float64`, you have to pass the two scalars

    ```SFF2(γ = 1.0, γ_noise = 1.0)```

# Reference(s)
[1] Bronasco, E., Laurent, A. B., & Huguet, B. (n.d.). High order integration of stochastic dynamics on Riemannian manifolds with frozen flow methods.
"""
@with_kw_noshow struct SFF2{T, Tmol, Tdmol} <: AbstractSDESamplerOrder2{T}
    "γ parameter of the diffusion process. It is related to the curvature of the streamline."
    γ::T = 1f0
    "parameter of the diffusion process to scale the variance."
    γ_noise::T = 1f0
    "mollifier."
    mollifier::Tmol = Base.Fix2(softplus, 10)
    "differential of mollifier."
    d_mollifier::Tdmol = Base.Fix2(∂softplus, 10)
    "Fixed time step?"
    adaptive::Bool = false
end
####################################################################################################
"""
$(TYPEDEF)

Structure to encode a cone to limit sampling the direction. 
This ensures that the angle in degrees between to consecutive streamline directions is less than `angle`.

The implemented condition is for `cn = Cone(angle)`.

```
(cn::Cone)(d1, d2) = dot(d1, d2) > cosd(cn.alpha)
```

## Fields

$(TYPEDFIELDS)

## Constructor

`Cone(angle)`
"""
struct Cone{𝒯 <: Real}
    "half section angle in degrees"
    alpha::𝒯
end
(cn::Cone)(d1, d2) = dot(d1, d2) > cosd(cn.alpha)

"""
$(TYPEDEF)

Model of streamlines.

# Internal fields (with default values):
$(TYPEDFIELDS)

# Methods
- `_apply_mask!(model, mask)` apply a mask to the raw SH tensor. See its doc string.
- `_getdata(model)` return the fodf data associated with the model.
- `size(model)` return `nx, ny, nz, nt`.
- `eltype(model)` return the scalar type of the data (default Float64).
- `get_lmax(model)` return the max `l` coordinate in of spherical harmonics.

# Constructors (pass the internal fields!)
- `Model()`
- `Model(Δt = 0.1f0)` for a Float32 Model
- `Model(Δt = 0.1, proba_min = 0.)` for a Float64 Model. You need to specify both fields `Δt` and `proba_min`
- `Model(odfdata = rand(10,10,10,45))` for custom ODF
"""
@with_kw_noshow struct Model{𝒯, 𝒯alg <: AbstractFODEvaluation, 𝒯d, 𝒯C, 𝒯mol}
    "Step size of the Model."
    Δt::𝒯 = 0.1f0
    "Spherical harmonics evaluation algorithm. Can be `PreComputeAllFOD()`, `DirectFOD()`."
    evaluation_algo::𝒯alg = PreComputeAllFOD()
    "ODF data, typically from nifti file but can be passed as an Array. Must be the list of ODF in the base of spherical harmonics. Hence, it should be an (abstract) 4d array."
    foddata::𝒯d = nothing
    "Cone function to restrict orientation sample. You can use a `Cone` or a custom function `(d1, d2) -> return_a_boolean`."
    cone::𝒯C = Cone(90f0)
    "Probability below which we stop tracking."
    proba_min::𝒯 = 0.0f0
    "Mollifier, used to make the fodf non negative. During odf evaluation, we effectively use `mollifier(fodf[angle,i,j,k])`."
    mollifier::𝒯mol = max_mollifier
end
@inline getdata(model::Model) = model.foddata
Base.size(model::Model) = size(getdata(model))
Base.eltype(model::Model{𝒯}) where 𝒯 = 𝒯
@inline get_lmax(model::Model) = get_lmax(getdata(model))
@inline get_basis(model::Model) = get_basis(getdata(model))
@inline get_evaluation(model::Model) = model.evaluation_algo

"""
$(TYPEDSIGNATURES)

`max(x, 0)` as mollifier to prevent negative ODF.
"""
max_mollifier(x) = max(0, x)
get_range(model::Model) = get_range(getdata(model))
get_array(model::Model) = _get_array(getdata(model))

function Base.show(io::IO, model::Model)
    printstyled(io, "Model with elype ", eltype(model), bold = true, color = :cyan)
    println(io, "\n ├─ Δt = ", model.Δt)
    println(io, " ├─ minimal probability     = ", model.proba_min)
    if model.cone isa Cone
        println(io, " ├─ cone                    = ", model.cone)
    end
    println(io, " ├─ mollifier               = ", model.mollifier)
    println(io, " ├─ evaluation of the basis = ", model.evaluation_algo)
    if model.foddata isa FODData
        println(io, " └─ data : (lmax = $(get_lmax(model)))")
        show(io, model.foddata; prefix = "      ")
    end
    if model.foddata isa AbstractArray
        println(io, " └─ data                = ", typeof(model.foddata))
    end
end

"""
$(TYPEDSIGNATURES)

Multiply the mask which is akin to a matrix of `Bool` with same size as the data stored in `model`. Basically, the mask `mask[ix, iy, iz]` ensures whether the voxel `(ix, iy, iz)` is discarded or not.

# Arguments

- `model::Model`.
- `mask` can be a `AbstractArray{3, Bool}` or a `NIVolume`.
"""
function _apply_mask!(model, mask)
    if ~isnothing(mask)
        nx, ny, nz, nsh = size(model)
        data = _get_array(getdata(model))
        for k = 1:nsh
            LV.@tturbo data[:, :, :, k] .*= mask
        end
    end
end
####################################################################################################
function save_streamlines end
