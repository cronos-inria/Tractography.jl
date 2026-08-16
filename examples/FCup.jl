using Revise, LinearAlgebra, GLMakie, NIfTI, StaticArrays
import Tractography as TG
Makie.inline!(true)
ENV["JULIA_DEBUG"] = TG

begin
foddata = TG.FODData("fod-FC.nii.gz")
# define the model for TMC
model = TG.Model(;Δt = 0.125f0,
            foddata,
            cone = TG.Cone(15f0),
            proba_min = 0.005f0,
            # mollifier = identity,
            )
            )

mask = NIfTI.niread("wm-FC.nii.gz");
# mask = NIfTI.niread("cross-wm.nii.gz")
TG._apply_mask!(model, mask);
end

begin
    f, sc = @time TG.plot_fod(model; n_sphere = 1500, radius = 0.3, st = 1);
    cam3d = Makie.cameracontrols(sc)
    cam3d.eyeposition[] = Vec3f(85, 95, -28)
    cam3d.lookat[] = Vec3f(84, 95, 59)
    rotate_cam!(sc.scene, 0, 0, -pi/2)
    f
end

# initial conditions for the streamlines
begin
Nmc = 100_000
_ind1 = findall(mask .== 1)
seed_id = rand(1:length(_ind1), Nmc)
seeds = zeros(Float32, 6, Nmc)
@time for i=1:Nmc
    seeds[1:3,i] .= TG.transform(model.foddata, _ind1[seed_id[i]])[SVector(1,2,3)]
    seeds[4:6,i] .= normalize(randn(SVector{3, Float64}))
end
end

alg = TG.Probabilistic();          kw = (;nt = 2000)
# alg = TG.Deterministic();          kw = (;nt = 2000)
# alg = TG.Transport();           kw = (;nt = 500_000, saveat = 125)
# alg = TG.Diffusion(γ = 0.99f0); kw = (;nt = 500_000, saveat = 125)
streamlines, stl_lengths = @time TG.sample(model, alg, seeds; kw...);

begin
    f = Figure(backgroundcolor = :white, size=(800,800)); sc = Axis(f[1,1], title = "$alg")
    @time TG.plot_fod!(sc, model; n_sphere = 500, radius = 0.3, st = 1);
    IND = findall(stl_lengths .> 400)
    TG.plot_streamlines!(sc, streamlines[1:3, :, IND])
    if sc isa LScene
        TG.add_frame!(sc)
        cam3d = Makie.cameracontrols(sc)
        cam3d.eyeposition[] = Vec3f(85, 95, -28)
        cam3d.lookat[] = Vec3f(84, 95, 59)
        rotate_cam!(sc.scene, 0, 0, -pi/2)
    end
    # sc.azimuth[]   = 0.0          # rotation autour de l'axe vertical (roll)
    # sc.elevation[] = deg2rad(85)  # proche de 90° => vue depuis le dessus
    # sc.zoom_mult[] = 1.0          # distance de la caméra
    f
end