# using Revise
using Test, LinearAlgebra
using Tractography
const TG = Tractography
using StaticArrays, NIfTI

let
    a = SMatrix{3,3}(Matrix{Float32}(Symmetric(rand(3,3))))
    av = TG.to_vec(a)
    @test a == TG.to_tensor(av)
end

function circular_field_dti(shape=(30, 30, 3); width_tensor = 2., Ty = 𝒯)
    κ = width_tensor
    # Generate sampling directions using Fibonacci sphere
    angles = TG.fibonacci_sampling(2000)
    bvectors = [SVector(TG.spherical_to_euclidean(d[1], d[2])...) for d in angles]
    
    # Initialize FOD values array
    dti_values = zeros(Ty, shape..., 6)
    center = [shape...] ./ 2 .- 0.5
    for i in axes(dti_values, 1), j in axes(dti_values, 2), k in axes(dti_values, 3) 
        x, y, z = @SVector[i, j, k] .- center
        r = sqrt(x * x + y * y)
        eigvec =
            @SMatrix [
                y / r  x / r  0;
                -x / r  y / r  0;
                0  0  1
            ]
        T = @SMatrix [1 0 0; 0 κ 0; 0 0 κ]
        tensor = eigvec * T * eigvec'
        dti_values[i, j, k, :] .= TG.to_vec(tensor)
    end
    mask = ones(Bool, size(dti_values)[1:3]...)
    # @error "Writing files..."
    # NIfTI.niwrite("dti.nii", NIfTI.NIVolume(dti_values))
    # # generate mask
    # NIfTI.niwrite("dti-wm.nii", NIfTI.NIVolume(mask))
    return dti_values, mask
end

let
    _dti, _mask = @time "dti" circular_field_dti((100, 100, 100); width_tensor = 3.0, Ty = Float32)
    
    𝒯 = Float32
    model = TG.Model(Δt = 𝒯(0.01),
                    foddata = TG.FODData(_dti, false; basis = TG.DTI()),
                    cone = TG.Cone(𝒯(15)),
    )
    show(model)
    TG._apply_mask!(model, _mask)
    seeds = zeros(𝒯, 6, 1)
    for i in axes(seeds, 2)
        X = LinRange(11, 11, size(seeds, 2))
        seeds[:,i] .= [X[i], 21, 15, 0, 1., 0]
    end
    TG.sample(model, TG.Deterministic(), seeds; nt = 8000, n_sphere = 401)

    model_d = TG.Model(Δt = 𝒯(0.0001),
                    foddata = TG.FODData(_dti, false; basis = TG.DTI()),
                    cone = TG.Cone(𝒯(15)),
                    evaluation_algo = TG.DirectFOD()
    )
    nt = 60000
    TG.sample(model_d, TG.Transport(;γ = 𝒯(1) ), 𝒯.(seeds); nt, maxfod_start = false);
    TG.sample(model_d, TG.Diffusion(;γ = 𝒯(1) ), 𝒯.(seeds); nt, maxfod_start = false);
end