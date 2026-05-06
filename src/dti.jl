function to_tensor(v_tensor::AbstractVector{𝒯}) where {𝒯}
    SA.SMatrix{3, 3, 𝒯, 9}( [v_tensor[1], v_tensor[4], v_tensor[5],
                             v_tensor[4], v_tensor[2], v_tensor[6],
                             v_tensor[5], v_tensor[6], v_tensor[3]] )
end

function to_vec(tensor::AbstractMatrix{𝒯}) where {𝒯}
    SA.SVector{6, 𝒯}(tensor[1,1], tensor[2,2], tensor[3,3], tensor[1,2], tensor[1,3], tensor[2,3])
end

@inline evaluate(::DTI, tensor, d) = LinearAlgebra.dot(d, tensor, d)

function _init(model::Model{𝒯, PreComputeAllFOD},
                alg::AbstractNotPureRejectionSampler,
                basis::DTI; 
                n_sphere = 400) where {𝒯}
    (;angles, directions) = _init_fibonacci(model, n_sphere)
    # Yₗₘ = get_vector_of_sh(angles, lmax)
    cone = isnothing(model.cone) ? nothing : [𝒯(model.cone(d1, d2)) for d1 in directions, d2 in directions]
    n_angles = length(angles)
    cache = ModelCache(; n_sphere = n_angles, Yₗₘ = nothing, dΩ = 𝒯(4pi / n_angles), angles, lmax = 0, cone, directions)
    # sample all FOD
    nx, ny, nz, _ = size(model)
    ni = 𝒯.(get_array(model))
    fod = zeros(𝒯, nx, ny, nz, n_angles)
    @time "DTI all FOD" for I in CartesianIndices(axes(fod))
        ni_v = @views ni[I[1], I[2], I[3], :]
        tensor = to_tensor(ni_v)
        d = SA.SVector(directions[I[4]]...)
        fod[I] = evaluate(basis, tensor, d)
    end
    @reset cache.odf = permutedims(fod, (4,1,2,3))
    return cache
end

function evaluate_for_diffusion(::DTI,
                                ::DirectFOD,
                                ϕ::𝒯,
                                θ::𝒯,
                                ::Int32,
                                voxels::Tuple{UInt32, UInt32, UInt32},
                                cache::ThreadedCache) where {𝒯}
    voxel_index₁, voxel_index₂, voxel_index₃ = voxels
    V = @view cache.odf[:, voxel_index₁, voxel_index₂, voxel_index₃]
    tensor = to_tensor(V)
    d = spherical_to_euclidean(θ, ϕ)
    F = evaluate(DTI(), tensor, SA.SVector(d))

    # local orthonormal basis
    st, ct = sincos(θ)
    sp, cp = sincos(ϕ)
    E₁ = SA.SVector(ct * cp, ct * sp, -st) # E_theta
    E₂ = SA.SVector(-sp, cp, 0)            # E_phi

    ∫F = one(𝒯)
    # Careful that ∂F∂ϕ = ∂(FODF)∂ϕ / sin(θ) to avoid division by zero that is why E₂ = (-sp, cp, 0) 
    # and not E₂ = (-sp*st, cp * st, 0) 
    ∂F∂θ = 2 * dot(d, tensor, E₁)
    ∂F∂ϕ = 2 * dot(d, tensor, E₂)

    return F, ∂F∂ϕ, ∂F∂θ, ∫F
end