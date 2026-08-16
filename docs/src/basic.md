# Simple example of tractography

This minimal example defines a `Model` from a fibre orientation distribution (fODF) stored in a NIfTI file.

```@example BASIC
import Tractography as TG

# define the Model as the streamline model
model = TG.Model(Δt = 0.125f0,
            foddata = TG.FODData((@__DIR__) * "/../../examples/cross-fod.nii.gz"),
            cone = TG.Cone(15f0),
            proba_min = 0.005f0,
            )
```
