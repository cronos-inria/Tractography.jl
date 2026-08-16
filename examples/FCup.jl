using Revise, LinearAlgebra, NIfTI, StaticArrays
import Tractography as TG
using GLMakie; Makie.inline!(true)
ENV["JULIA_DEBUG"] = TG

begin
foddata = TG.FODData("fod-FC.nii.gz")
# define the model for TMC
model = TG.Model(;Δt = 0.125f0,
            foddata,
            cone = TG.Cone(15f0),
            proba_min = 0.001f0,
            # mollifier = identity,
            )
            )

mask = NIfTI.niread("wm-FC.nii.gz");
TG._apply_mask!(model, mask); TG._apply_mask!(model_d, mask);
end

# initial conditions for the streamlines
begin
Nmc = 10
seeds = TG.from_fod(model, Nmc)
@time for i=1:Nmc
    seeds[3,i] = 3
end
end

alg = TG.Probabilistic();          kw = (;nt = 2000)
# alg = TG.Deterministic();          kw = (;nt = 2000)
# alg = TG.Transport(adaptive = false); kw = (;nt = 100_000, saveat = 50)
# alg = TG.Diffusion(γ = 1f0, γ_noise = 0.1f0, adaptive = false); kw = (;nt = 100_000, saveat = 50)
streamlines, stl_lengths = @time TG.sample(
    (alg isa TG.AbstractSDESampler ? model_d : model), 
            alg, 
            seeds; 
            kw...,
            );

begin
    f = Figure(backgroundcolor = :white, size=(800,800)); 
    sc = Axis(f[1,1], title = "$alg")
    # sc = LScene(f[1,1])
    @time TG.plot_fod!(sc, model; n_sphere = 1500, radius = 0.2, st = 1);
    IND = findall(stl_lengths .> 10)
    TG.plot_streamlines!(sc, streamlines[1:3, :, IND], stl_lengths[IND])
    scatter!(sc, streamlines[1:3, 1, IND])
    # xlims!(sc, (120,140));ylims!(sc, (60,80))

    if sc isa LScene
        TG.add_frame!(sc)
        cam3d = Makie.cameracontrols(sc)
        cam3d.eyeposition[] = Vec3f(85, 95, -28)
        cam3d.lookat[] = Vec3f(84, 95, 59)
        rotate_cam!(sc.scene, 0, 0, -pi/2)
    else
        sc.title[] = "$(typeof(alg).name.name) - $(length(IND)) / $(length(stl_lengths)) selected streamlines"
    end
    f
#############
# begin
#     f, sc = @time TG.plot_fod(model; n_sphere = 1500, radius = 0.3, st = 1);
#     cam3d = Makie.cameracontrols(sc)
#     cam3d.eyeposition[] = Vec3f(85, 95, -28)
#     cam3d.lookat[] = Vec3f(84, 95, 59)
#     rotate_cam!(sc.scene, 0, 0, -pi/2)
#     f
# end
