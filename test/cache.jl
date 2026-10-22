using Test, LinearAlgebra, Accessors
using Tractography
const TG = Tractography

let
    model = Model(Δt = 0.125f0,
                foddata = TG.FODData("../examples/fod-FC.nii.gz"),
                cone = Cone(45f0),
                )

    alg = Deterministic()
    cache_c = TG._init(model, Probabilistic(), TG.get_basis(model); n_sphere = 1000)
    @test cache_c.dΩ ≈ (4pi) / (cache_c.n_sphere)

    @test all(isfinite, cache_c.odf)

    # test norm of directions
    @test all(x -> abs(norm(x) - 1) < 1e-6, eachrow(cache_c.directions))

    # test spherical coordinates
    @test all(x -> (0 <= x[1] <= Float32(pi)),  cache_c.angles)
    @test all(x -> (0 <= x[2] <= 2pi), cache_c.angles)

    # test normalisation
    sum(cache_c.odf, dims = 1) * cache_c.dΩ |> unique |> sort
end

# the Fibonacci lattice spans [0, π] in θ (both poles are included)
let
    angles = TG.fibonacci_sampling(400, Float64)
    @test length(angles) == 401
    @test all(x -> 0 <= x[1] <= pi, angles)
    @test all(x -> 0 <= x[2] < 2pi, angles)
    @test any(x -> x[1] == 0, angles) && any(x -> x[1] ≈ pi, angles)
end
end