using Test, Tractography, LinearAlgebra, Accessors
const TG = Tractography

let
    TG.PlottingFOD()
    TG.PreComputeAllFOD()
    TG.DirectFOD()
    
    TG.softplus(0, 1)
    TG.∂softplus(0, 1)
    
    TG.spherical_to_euclidean(0,0)
    @test all(TG.euclidean_to_spherical(TG.spherical_to_euclidean(0.1, -0.01)...) .≈ [0.1, -0.01])
    @test all(TG.euclidean_to_spherical(TG.spherical_to_euclidean(0.1, 3pi/2)...) .≈ [0.1, -pi/2])
    @test all(TG.euclidean_to_spherical(TG.spherical_to_euclidean(0.1, -pi/2)...) .≈ [0.1, -pi/2])
    u0 = normalize(rand(3))
    @test all([TG.spherical_to_euclidean(TG.euclidean_to_spherical(u0...)...)...] .≈ u0)
    
    
    p0 = normalize(rand(3))
    v0 = normalize(rand(3)); v0 .-= dot(p0,v0) .* p0
    @test dot(v0, p0) ≈ 0 atol = 1e-14
    u = TG.Exp𝕊²(p0, v0, 0.2)
    @test norm(u) ≈ 1
end

for eval_alg in (TG.PreComputeAllFOD(),)
    model = TG.Model(Δt = 0.125f0,
                foddata = TG.FODData((@__DIR__) * "/../examples/fod-FC.nii.gz"),
                cone = TG.Cone(15),
                proba_min = 0.005f0,
                evaluation_algo = eval_alg,
                )

    TG._apply_mask!(model, ones(Float32, 64, 64, 3))

    show(stdout, model)

    TG.sample(model, TG.Deterministic(), rand(Float32, 6, 2); nt = 10, maxfod_start = true, reverse_direction = true);
    TG.sample(model, TG.Connectivity(TG.Deterministic()), rand(Float32, 6, 2); nt = 10, maxfod_start = true);
    TG.sample(model, TG.Probabilistic(), rand(Float32, 6, 2); nt = 10);
    TG.sample(model, TG.Connectivity(TG.Probabilistic()), rand(Float32, 6, 2); nt = 10);
end

for eval_alg in (TG.DirectFOD(), TG.PreComputeAllFOD())
    model_diffusion = TG.Model(Δt = 0.001f0,
                foddata = TG.FODData((@__DIR__) * "/../examples/fod-FC.nii.gz"),
                proba_min = 0.0f0,
                evaluation_algo = eval_alg,
                )
    TG.sample(model_diffusion, TG.Transport(), rand(Float32, 6, 2); nt = 10, maxfod_start = true, reverse_direction = true);
    TG.sample(model_diffusion, TG.Diffusion(), rand(Float32, 6, 2); nt = 10, maxfod_start = true, reverse_direction = true);
    TG.sample(model_diffusion, TG.Connectivity(TG.Diffusion()), rand(Float32, 6, 2); nt = 10, maxfod_start = true, reverse_direction = true);
    TG.sample(model_diffusion, TG.Diffusion(adaptive = false), rand(Float32, 6, 2); nt = 10);

    # cache diffusion
    show(TG.Diffusion())
    show(TG.Transport())
    cache = TG.init(model_diffusion, TG.Diffusion())
    cache = TG.init((@set model_diffusion.evaluation_algo = TG.PreComputeAllFOD()), TG.Diffusion())
    TG.Exp𝕊²(rand(3), zeros(3), 1)
end
########################
# cache
let
    model = TG.Model(Δt = 0.125f0,
    foddata = TG.FODData((@__DIR__) * "/../examples/fod-FC.nii.gz"),
    cone = TG.Cone(15),
    proba_min = 0.005f0,
    )
    Nmc = 10
    seeds = rand(Float32, 6, Nmc)
    streamlines = zeros(Float32, 6, 20, Nmc)
    tract_length = zeros(UInt32, Nmc)
    alg = Probabilistic()
    cache = TG.init(model, alg)
    TG.get_angles(cache, 1)
    show(stdout, cache)
    cache = TG._init(model, alg, TG.get_basis(model))
    show(stdout, cache)
    cache = TG.init(model, TG.Diffusion())
    show(stdout, cache)
    
    TG._init((@set model.evaluation_algo = TG.PlottingFOD()), TG.Deterministic(), TG.get_basis(model))
end
########################
# test the integral of spherical harmonics computed with fibonacci sampling
let 
    my_ylm(t,p,l,m) = TG.ro_sh(t,p,l,m)
    N = 5000
    angles = TG.fibonacci_sampling(N)
    dΩ = 4π / N
    s  = sum(my_ylm(θ, ϕ, 0, 0) for (θ, ϕ) in angles) * dΩ
    @test s ≈ sqrt(4π) atol=sqrt(1/N)
    println("Y₀₀: $s  (attendu √4π = $(sqrt(4π)))")
    s2 = sum(my_ylm(θ, ϕ, 2, 1) for (θ, ϕ) in angles) * dΩ
    @test s2 ≈ 0 atol=sqrt(1/N)
    println("Y₂,1: $s2  (attendu 0)")
    s3 = sum(my_ylm(θ, ϕ, 11, 2) for (θ, ϕ) in angles) * dΩ
    @test s3 ≈ 0 atol=sqrt(1/N)
    println("Y11,2: $s3  (attendu 0)")
    s4 = sum(my_ylm(θ, ϕ, 22, 2) for (θ, ϕ) in angles) * dΩ
    @test s4 ≈ 0 atol=sqrt(1/N)
    println("Y22,2: $s4  (attendu 0)")
end