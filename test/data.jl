using Tractography, StaticArrays
const TG = Tractography

TG.FODData(rand(10,10,10,10), false)

model = TG.Model(Δt = 0.1, proba_min = 0.0,
            foddata = TG.FODData((@__DIR__) * "/../examples/fod-FC.nii.gz",
            basis = TG.DTI()
            )
)

model = TG.Model(Δt = 0.1, proba_min = 0.0,
            foddata = TG.FODData((@__DIR__) * "/../examples/fod-FC.nii.gz")
)
TG.issymmetric(model.foddata.basis)

show(stdout, model.foddata; full = true)

x = SVector(rand(3)...)
y = model.foddata.transform.S * SVector(x..., 1)
TG.transform(model.foddata, CartesianIndex{3}(1,1,1))
@test TG.transform(model.foddata, x) == y
@test TG.transform_inv(model.foddata, y)[1:3] ≈ x rtol = 1e-6

TG.get_range(model.foddata)
TG._get_array(zeros(2))
TG._my_typeof(1)

TG.from_fod(model, 10; maxfod_start = false)
TG.from_fod(model, 10; maxfod_start = true)
mask = model.foddata.data[:,:,:,1].>0;
TG.from_mask(model, mask, 10)