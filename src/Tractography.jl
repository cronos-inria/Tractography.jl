module Tractography
    using DocStringExtensions
    using Accessors
    using NIfTI
    using Parameters, LinearAlgebra
    import StaticArrays as SA
    import LoopVectorization as LV
    import FastTransforms
    using Random

    # sampling method of FODF
    export DirectFOD, PreComputeAllFOD
    export FODData, TMC, Cone, sample, init
    export Probabilistic, Deterministic, Diffusion, Connectivity
    export save_streamlines

    # plotting
    export plot_streamlines!, plot_fod, plot_fod!

    include("plot.jl")
    include("models.jl")
    include("seeds.jl")
    include("data.jl")
    include("utils.jl")
    include("sph.jl")
    include("tmccache.jl")
    include("sample.jl")
    include("gpu.jl")
    include("diffusion.jl")
end
