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
    tf = model.foddata.transform

    non_zeros = @views findall(x -> x>0, odfs[:, :, :, 1])
    n_zeros = length(non_zeros)

    cache = _init_fibonacci_and_sph(model, n_sphere)
    odf = zeros(Float64, n_sphere+1)
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
        if maxfod_start
            ind_u = argmax(odf)
        else
            t = rand(𝒯) * 𝒯(sum(odf))
            cw = zero(𝒯)
            ind_u = 0
            for nₐ in eachindex(odf)
                cw += 𝒯(odf[nₐ])
                if cw >= t
                    ind_u = nₐ
                    break
                end
            end
        end
        seeds[4:6, i] .= directions[ind_u]
    end
    return seeds
end

"""
$(TYPEDSIGNATURES)

Generate seeds from a 3D mask. The seeds are generated randomly inside voxels with non-zero values in the mask. The orientations are random.
"""
function from_mask(model::Model{𝒯}, mask::AbstractArray{Bool, 3}, n_seeds::Int) where {𝒯}
    seeds = zeros(𝒯, 6, n_seeds)
    non_zeros = findall(mask)
    n_zeros = length(non_zeros)
    tf = model.foddata.transform

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
        # sample direction
        randn!(seeds[4:6, i])
        normalize!(seeds[4:6, i])
    end
    return seeds
end