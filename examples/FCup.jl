using Revise, LinearAlgebra, GLMakie, NIfTI
using Tractography
const TG = Tractography
Makie.inline!(true)
ENV["JULIA_DEBUG"] = Tractography

# define the model for TMC
model = TG.Model(Δt = 0.125f0,
            foddata = FODData("fod-FC.nii.gz"),
            # odfdata = ODFData("cross-fod.nii.gz"),
            cone = Cone(15f0),
            proba_min = 0.005f0,
            )

mask = NIfTI.niread("wm-FC.nii.gz");
# mask = NIfTI.niread("cross-wm.nii.gz")
TG._apply_mask!(model, mask);

begin
    f, sc = @time TG.plot_fod(model; n_sphere = 1500, radius = 0.3, st = 1);
    cam3d = Makie.cameracontrols(sc)
    cam3d.eyeposition[] = Vec3f(85, 95, -28)
    cam3d.lookat[] = Vec3f(84, 95, 59)
    rotate_cam!(sc.scene, 0, 0, -pi/2)
    f
end

# initial conditions for the streamlines
Nmc = 100_000
_ind1 = findall(mask .== 1)
seed_id = rand(1:length(_ind1), Nmc)
seeds = zeros(Float32, 6, Nmc)
for i=1:Nmc
    seeds[:,i] .= vcat(TG.transform(model.foddata, _ind1[seed_id[i]])[1:3]..., normalize(randn(3)))
end

# cache = TG.init(model, Deterministic(); n_sphere = 1000)
streamlines, stl_lengths = @time TG.sample(model, Deterministic(), seeds; nt = 1000);

begin
    f, sc = @time TG.plot_fod(model; n_sphere = 500, radius = 0.3);
    plot_streamlines!(sc, streamlines[1:3, :, :])
    f
end