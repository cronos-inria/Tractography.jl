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
    export Probabilistic, Deterministic, Diffusion, Connectivity
    export FODData, Model, Cone, sample, init
    export save_streamlines

    # plotting
    export plot_streamlines!, plot_fod, plot_fod!

    import Adapt

    include("plot.jl")
    include("models.jl")
    include("seeds.jl")
    include("data.jl")
    include("utils.jl")
    include("modelcache.jl")
    include("sph.jl")
    include("dti.jl")
    include("sample.jl")
    include("gpu.jl")
    include("diffusion.jl")
end
