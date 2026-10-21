function _init_fibonacci(model::Model{𝒯}, n_sphere) where {𝒯}
    angles = fibonacci_sampling(n_sphere, 𝒯)
    directions = [spherical_to_euclidean(d[1], d[2]) for d in angles]
    (;angles, directions)
end

function _init_fibonacci_and_sph(model::Model{𝒯}, n_sphere) where {𝒯}
    (;angles, directions) = _init_fibonacci(model, n_sphere)
    n_angles = length(angles)
    lmax = get_lmax(model)
    Yₗₘ = get_vector_of_sh(angles, lmax)
    cone = isnothing(model.cone) ? nothing : [𝒯(model.cone(d1, d2)) for d1 in directions, d2 in directions]
    ModelCache(; n_sphere = n_angles, Yₗₘ, dΩ = 𝒯(4pi / n_angles), angles, lmax, cone, directions)
end

# for plotting
function _init(model::Model{𝒯, PlottingFOD},
                alg::AbstractNotPureRejectionSampler,
                basis::SphericalHarmonics;
                n_sphere = 400) where {𝒯}
    _init_fibonacci_and_sph(model, n_sphere)
end

function _init(model::Model{𝒯, PreComputeAllFOD},
                alg::AbstractNotPureRejectionSampler,
                basis::SphericalHarmonics; 
                n_sphere = 400) where {𝒯}
    cache = _init_fibonacci_and_sph(model, n_sphere)
    _build_cache_from_Y_matrix(model, cache, cache.Yₗₘ)
end

function _build_cache_from_Y_matrix(model::Model{𝒯}, cache, Ysv) where {𝒯}
    na = size(Ysv, 1)
    # compute all FOD
    nx, ny, nz, nt = size(model)
    # we optimize the array layout because loops occur on the angle variable
    # and julia is column major
    ni = 𝒯.(get_array(model))
    ni_v = reshape(ni, nx*ny*nz, nt)
    odf_v  = @time_debug "Mat-Vec:" ni_v * Ysv'; # vector view
    odf = reshape(odf_v, nx, ny, nz, na);
    @time_debug "Mollifier:" LV.@tturbo @. odf = model.mollifier(odf)
    @reset cache.odf = permutedims(odf, (4,1,2,3))
    return cache
end

"""
$(TYPEDSIGNATURES)

Sample the Model `model`.

## Arguments
- `model::Model`
- `alg` sampling algorithm, `Deterministic, Probabilistic, Diffusion, etc`.
- `seeds` matrix of size `6 x Nmc` where `Nmc` is the number of Monte-Carlo simulations to be performed.
- `mask = nothing` matrix of boolean where to stop computation. See also `_apply_mask`.

## Optional arguments
- `nt::Int` maximal number of steps to compute each streamline.
- `n_sphere::Int = 400` number of points to discretize the sphere on which we evaluate the spherical harmonics.
- `maxfod_start::Bool` for each locations, use direction provided by the argmax of the ODF.
- `reverse_direction::Bool` reverse initial direction.
- `nthreads::Int = 8` number of threads on CPU.
- `gputhreads::Int = 512` number of threads on GPU.
- `saveat::Int = 1` store one point every `saveat` integration steps. Only available for `alg <: AbstractSDESampler` (i.e. `Diffusion`/`Transport`); the `Probabilistic`/`Deterministic` path errors if `saveat > 1`. It reduces the second dimension of `streamlines` and the values returned in `streamlines_length` (which then counts recorded points, not integration steps), but does not change the integrated trajectory, so it does not change the physical length; with `adaptive = true` the step size varies and cannot be recovered from `saveat` alone.

## Output
- `streamlines` with shape `3 x n_saved x Nmc`, where `n_saved` depends on the
  algorithm and on `saveat`:
  - `saveat = 1`: a point is stored at every integration step, so `n_saved = nt`;
  - `saveat > 1` (only for `alg <: AbstractSDESampler`, i.e. `Diffusion`/`Transport`):
    one point is stored every `saveat` integration steps, so for a streamline
    followed to its end `n_saved = 1 + (nt - 1) ÷ saveat`;
  - `Endpoints` mode: only the first and last points are kept, so the second
    dimension is `2` and `n_saved = 2`.
- `streamlines_length::Vector{UInt32}` of length `Nmc`. Its entries are a count,
  **not a physical length (mm)**, and their meaning depends on the algorithm:
  - `Probabilistic`/`Deterministic`: number of points stored for each streamline
    (the number of integration steps, seed included). `saveat > 1` is not
    supported on this path (it errors).
  - `Diffusion`/`Transport` (full streamlines): number of points actually stored,
    i.e. `1 + (nt - 1) ÷ saveat` for a streamline followed to its end. This is
    directly the last valid index along the second dimension of `streamlines`,
    so it can be used to slice or plot each streamline.
  - `Endpoints`: number of integration steps (divided by `saveat`), which is not
    an index into `streamlines` (only the start/end points are stored).
  `saveat` therefore reduces `streamlines_length` on the `Diffusion`/`Transport`
  paths: it counts *recorded points*, not integration steps. Because the step
  size may vary (`adaptive = true`), it cannot be converted to a distance by a
  single multiplication.
"""
function sample(model::Model{𝒯},
                alg::AbstractSampler,
                seeds::AbstractMatrix{𝒯},
                mask::Union{Nothing, AbstractArray{Bool}} = nothing;
                nt::Int = 1000,
                n_sphere::Int = 400,
                maxfod_start::Bool = false,
                reverse_direction::Bool = false,
                nthreads::Int = 8,
                gputhreads::Int = 512,
                saveat::Int = 1,
                ) where {𝒯}
    if size(seeds, 1) != 6
        error("The seeds dimension must be 6 x n_seed")
    end
    streamlines = zeros(𝒯, 3, (alg isa Endpoints ? 2 : 1 + div(nt - 1, saveat)), size(seeds, 2))
    streamlines_length = zeros(UInt32, size(seeds, 2))
    cache = init(model, alg; n_sphere)
    if ~isnothing(mask)
        _apply_mask!(model, mask)
    end
    sample!(streamlines,
            streamlines_length,
            model,
            cache,
            alg,
            seeds;
            maxfod_start,
            reverse_direction,
            nthreads,
            gputhreads,
            saveat,
            nₜ = nt)
    return streamlines, streamlines_length
end