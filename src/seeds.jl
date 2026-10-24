ispositive(x) = x > 0

"""
$(TYPEDSIGNATURES)

Generate seeds from orientation distribution functions.

## Keyword arguments
- `maxfod_start = false` The seeds are generated in voxels with non-zero average FOD with the orientations importance sampled.
- `maxfod_start = true` The seeds are generated in voxels with non-zero average FOD with the orientations corresponding to the maximum probability.
"""
function from_fod(model::Model{𝒯}, n_seeds::Int; n_sphere = 1000, maxfod_start = false) where {𝒯}
    seeds = zeros(𝒯, 6, n_seeds)
    odfs = _get_array(model.foddata)
    tf = _get_transform(model)

    non_zeros = @views findall(ispositive, odfs[:, :, :, 1])
    n_zeros = length(non_zeros)

    cache = _init_fibonacci_and_sph(model, n_sphere)
    odf = zeros(𝒯, n_sphere+1)
    directions = cache.directions

    if n_zeros == 0
        error("All fODFs are zero!")
    end

    for i in axes(seeds, 2)
        index = rand(1:n_zeros)
        I = non_zeros[index]
        position =  transform(tf, I)
        for k = 1:3
            seeds[k, i] = position[k] + rand(𝒯) - 𝒯(1//2)
        end
        # sample direction, slow because not row major
        @views mul!(odf, cache.Yₗₘ, odfs[I[1], I[2], I[3], :])
        ind_u = _sample_direction_index(odf, maxfod_start, 𝒯)
        seeds[4:6, i] .= directions[ind_u]
    end
    return seeds
end

"""
$(TYPEDSIGNATURES)

Return the index of the sampled direction in the fODF vector `odf`:
- `maxfod_start = true`: direction with the largest fODF value (`argmax`);
- otherwise: direction drawn from the fODF considered as a (unnormalized) probability distribution.
"""
function _sample_direction_index(odf, maxfod_start::Bool, 𝒯)
    if maxfod_start
        return argmax(odf)
    end
    t = rand(𝒯) * 𝒯(sum(odf))
    cw = zero(𝒯)
    for nₐ in eachindex(odf)
        cw += 𝒯(odf[nₐ])
        cw >= t && return nₐ
    end
    return lastindex(odf)
end

"""
$(TYPEDSIGNATURES)

Resample the directions (`seeds[4:6, :]`) of the pre-positioned `seeds` from the fODF, keeping the positions (`seeds[1:3, :]`, in native space) unchanged. For each position, the voxel is obtained by inverting the affine transform and clamped to the volume.
"""
function _set_seeds_from_fod!(model::Model{𝒯}, seeds::AbstractMatrix{𝒯}; n_sphere = 1000, maxfod_start = false) where {𝒯}
    size(seeds, 1) == 6 || error("The seeds must be a 6 x n_seeds matrix")
    odfs = _get_array(model.foddata)
    tf = _get_transform(model)
    nx, ny, nz, _ = size(model)

    cache = _init_fibonacci_and_sph(model, n_sphere)
    odf = zeros(𝒯, n_sphere + 1)
    directions = cache.directions

    for i in axes(seeds, 2)
        I = get_voxel_index(tf, (seeds[1, i], seeds[2, i], seeds[3, i]))
        I = (clamp(Int(I[1]), 1, nx), clamp(Int(I[2]), 1, ny), clamp(Int(I[3]), 1, nz))
        @views mul!(odf, cache.Yₗₘ, odfs[I[1], I[2], I[3], :])
        ind_u = _sample_direction_index(odf, maxfod_start, 𝒯)
        seeds[4:6, i] .= directions[ind_u]
    end
    return seeds
end

"""
$(TYPEDSIGNATURES)

Generate `n_seeds` seeds from the 3D binary mask `mask`.

For each seed, a voxel with `mask[I] == true` is drawn uniformly at random,
with replacement. The seed position is the image of the voxel centre by the
affine transform of `model.foddata`, plus a random offset
`sigma * (rand(𝒯) - 1/2)` in each native coordinate. The orientation is a
random unit vector, uniformly distributed on the sphere; it is not sampled
from the fODF (see [`from_fod`](@ref) for fODF-informed orientations).

Voxels are drawn with replacement, so a voxel can receive several seeds and
`n_seeds` may exceed `count(mask)`. `mask` must share the voxel grid of
`model.foddata`.

## Arguments
- `model::Model`: provides the affine transform `model.foddata.transform`.
- `mask::AbstractArray{Bool, 3}`: binary mask; only `true` voxels are seeded.
- `n_seeds::Int`: number of seeds to generate.

## Keyword arguments
- `sigma = one(𝒯)`: scale of the random offset around the voxel centre, in
  native coordinates. `sigma = 0` puts the seeds exactly at voxel centres;
  `sigma = 1` spreads them over one unit around the centre for unit voxel size.
  The offset does not account for anisotropic voxel sizes or rotations of the
  affine transform.

## Output
A `6 x n_seeds` matrix of element type `𝒯`: rows `1:3` are the positions in
native coordinates, rows `4:6` a unit orientation vector.

## Errors
Throws an error if `mask` contains no `true` voxel.

## Notes
The result depends on the global random number generator; use `Random.seed!`
for reproducibility.
"""
function from_mask(model::Model{𝒯}, mask::AbstractArray{Bool, 3}, n_seeds::Int; sigma = one(𝒯)) where {𝒯}
    tf = model.foddata.transform
    return from_mask(tf, mask, n_seeds, 𝒯; sigma)
end

"""
$(TYPEDSIGNATURES)

Same as `from_mask(model, mask, n_seeds; sigma)`, but with the affine transform
`tf` passed directly.

## Arguments
- `tf::Transform`: affine transform from voxel to native coordinates.
- `mask::AbstractArray{Bool, 3}`: binary mask; only `true` voxels are seeded.
- `n_seeds::Int`: number of seeds to generate.
- `𝒯 = Float32`: element type of the returned matrix. The `Model` method passes
  the element type of the model.

## Keyword arguments
- `sigma = one(𝒯)`: see [`from_mask`](@ref).
"""
@views function from_mask(tf::Transform, mask::AbstractArray{Bool, 3}, n_seeds::Int, 𝒯 = Float32; sigma = one(𝒯))
    seeds = zeros(𝒯, 6, n_seeds)
    non_zeros = findall(mask)
    n_zeros = length(non_zeros)

    if n_zeros == 0
        error("All mask values are zero!")
    end

    for i in axes(seeds, 2)
        index = rand(1:n_zeros)
        I = non_zeros[index]
        position = transform(tf, I)
        for k = 1:3
            seeds[k, i] = position[k] + (rand(𝒯) - 𝒯(1//2)) * sigma
        end
        # sample direction
        randn!(seeds[4:6, i])
        normalize!(seeds[4:6, i])
    end
    return seeds
end