# Endpoints estimate

```@contents
Pages = ["endpoints.md"]
Depth = 3
```

The algorithm [`Tractography.Endpoints`](@ref) allows to estimate structural connectivity. In short, 

```julia
import Tractography as TG

alg = TG.Endpoints(TG.Deterministic())
streamlines, tract_length = TG.sample(model, alg, seeds);
```

returns the first/last point on the streamline. You are then required to map both points with an atlas.