# Tractography.jl

`Tractography.jl` is a high-performance Julia package for brain tractography that leverages **parallel computing and specialized hardware (e.g., GPUs)** to reconstruct white matter fiber bundles from diffusion-weighted MRI data. This enables researchers to study the structural connectivity of the brain at unprecedented scales.

![](brain.png)

## Key Features

- **GPU acceleration**: Massive parallelization for processing billions of streamlines
- **Multiple tracking algorithms**: Including stochastic methods
- **Flexible seeding**: Generate seeds from FOD data or anatomical masks
- **Visualization**: Built-in plotting recipes for Makie.jl
- **Export capabilities**: Save tractograms to standard TCK format
- **High performance**: Successfully used to sample 500 billion streamlines on GPU

## 📦 Installation

Assuming you have Julia installed, add `Tractography.jl` using the package manager:

```julia
import Pkg
Pkg.add("Tractography")
```

## 📚 Citing this work

Citation instructions will be added soon.

## 🧑‍💻 Related Software

Similar algorithms are implemented in the Python package:

- [Cronos Tractography](https://gitlab.inria.fr/cronos/software/tractography)

Several excellent tractography software packages are available and listed on the [IST website](https://github.com/International-Society-for-Tractography/ist-tractography-db?tab=readme-ov-file#tractography--diffusion-software):

- **[DIPY](https://dipy.org)** - Comprehensive diffusion MRI analysis in Python
- **[DSI Studio](https://dsi-studio.labsolver.org)** - Diffusion MRI analysis tool
- **[Entrack](https://vitalab.github.io/article/2019/11/21/entrack.html)** - Deep learning-based tractography
- **[ExploreDTI](https://www.exploredti.com)** - DTI and HARDI analysis toolbox
- **[MRtrix3](https://github.com/MRtrix3/mrtrix3)** - Leading software for diffusion MRI analysis (lacks GPU support)
- **[Scilpy](https://github.com/scilus/scilpy)** - Python tools for diffusion MRI processing
- **[Trekker](https://dmritrekker.github.io)** - Fast parallel tractography

### Julia Ecosystem

`Tractography.jl` is currently the only Julia package focused specifically on tractography. Related Julia packages include:

- **[Fibers.jl](https://github.com/lincbrain/Fibers.jl)** - Tools for diffusion MRI data
- **[Microstructure.jl](https://github.com/Tinggong/Microstructure.jl)** - Microstructure modeling
- **[NeuroFormats.jl](https://github.com/dfsp-spirit/NeuroFormats.jl)** - Neuroimaging file format support
- **[JuliaNeuroscience](https://github.com/JuliaNeuroscience)** - Neuroscience tools organization

## Performance

The examples in this documentation prioritize **clarity and simplicity** over maximum performance. However, `Tractography.jl` is capable of extreme-scale tractography.

### Proven At Scale

This package was used to sample **500 billion streamlines on GPU** for a recent publication:

- Yanis Aeschlimann, Samuel Deslauriers-Gauthier, Romain Veltz. **GPU tractography: What can we learn from half a trillion streamlines?** International Society for Tractography Conference - IST 2025, Oct 2025, Bordeaux, France. [⟨hal-05272265⟩](https://inria.hal.science/hal-05272265v1)

This demonstrates the package's capability to handle production-scale neuroimaging research with GPU acceleration.
