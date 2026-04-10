using Tractography
using Test

@testset "Tractography.jl" begin
    include("basic-sampling.jl")
    include("cache.jl")
    include("data.jl")
    include("sh.jl")
    include("plot.jl")
    include("test_drift_expression.jl")
end
