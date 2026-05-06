using Tractography, StaticArrays, Test
const TG = Tractography

let
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
    @test TG._my_typeof(1) == Int64

    TG.from_fod(model, 10; maxfod_start = false)
    TG.from_fod(model, 10; maxfod_start = true)
    mask = model.foddata.data[:,:,:,1].>0;
    TG.from_mask(model, mask, 10)

    # add a mask and test normalization of FODF, ∫FODF = 1?
    # Carefull because the mollifier "destroys" this normalization
    model = TG.Model(Δt = 0.1, proba_min = 0.0,
                foddata = TG.FODData((@__DIR__) * "/../examples/fod-FC.nii.gz"; normalize_it = false),
                mollifier = identity,
    )
    _mask = model.foddata.data.raw[:,:,:,1] .> 0.2
    TG._apply_mask!(model, _mask)
    TG._normalize_data!(model.foddata.data)
    @test TG.is_normalized(model.foddata) == false
    @test sort(unique(model.foddata.data.raw[:,:,1,1]) ) == [0,1]
    N_a = 10000
    cache = TG.init(model, TG.Deterministic(), n_sphere = N_a)
    # test les valeurs de l'integrale
    @test all(x-> abs(x-1)< sqrt(1/N_a) , (sum(cache.odf[:,:,:,1], dims=1) * cache.dΩ / sqrt(4pi) |> unique |> sort)[2:end])
end
