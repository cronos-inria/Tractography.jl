using Tractography
using Test

@testset "Tractography.jl" begin
    @testset "basic sampling" begin
        include("basic-sampling.jl")
    end
    @testset "cache" begin
        include("cache.jl")
    end
    @testset "data" begin
        include("data.jl")
    end
    @testset "spherical harmonics" begin
        include("sh.jl")
    end
    @testset "plot" begin
        include("plot.jl")
    end
    @testset "drift expression" begin
        include("test_drift_expression.jl")
    end
    @testset "DTI" begin
        include("dti.jl")
    end
end
