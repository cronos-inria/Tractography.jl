abstract type Abstract_fODFBasis end
abstract type AbstractInterpolation end

"""
The fODF are specified in the Spherical Harmonics basis.
If `issymmetric == true`, only the even `l` are taken into account to yield symmetric fODF.
"""
struct SphericalHarmonics{issymmetric} <: Abstract_fODFBasis end
issymmetric(::SphericalHarmonics{_issymmetric}) where {_issymmetric} = _issymmetric

"""
The fODF are specified in the Diffusion Tensor Imaging basis. They are therefore expressed as

f(u) = < u, D⋅u > 

where D is a real symmetric positive definite 3 x 3 matrix.
"""
struct DTI <: Abstract_fODFBasis end
##########################################################################################
struct NearestNeighbor <: AbstractInterpolation end
##########################################################################################
"""
$(TYPEDEF)

Structure to hold the affine transform from the real world to voxel coordinates.

# Internal fields
$(TYPEDFIELDS)

# Methods
- `transform(tf::Transform, x)` return the linear transform `tf.S * x`
- `transform_inv(tf::Transform, x)` return the inverse linear transform `tf.Sinv * x`

# Constructor
- `Transform(I(4), zeros(3))`. For performance reasons, you should pass static arrays, for example using `StaticArrays.jl`.
"""
struct Transform{𝒯s, 𝒯t}
    "Forward linear transform."
    S::𝒯s
    "Inverse linear transform."
    Sinv::𝒯s
    "Translation."
    T::𝒯t
end
@inline transform(tf::Transform, x) = tf.S * x # for plotting
@inline transform(tf::Transform, x::CartesianIndex{3}) = transform(tf, SA.SVector(Tuple(x)..., 1)) # for plotting
@inline transform(tf::Transform, x::SA.SVector{3}) = transform(tf, SA.SVector(x..., 1)) # for plotting
@inline transform_inv(tf::Transform, x) = tf.Sinv * x

function Transform(S::AbstractArray, T::AbstractVector)
    if eltype(S) != eltype(T)
        @warn("Careful! Not all arrays have the same element type!")
    end
    Transform(S, pinv(S), T)
end
##########################################################################################
"""
$(TYPEDEF)

Structure to hold FOD data.

# Internal fields
$(TYPEDFIELDS)

# Methods
- `get_lmax(::FODData)` returns the max `l` coordinate in of spherical harmonics.
- `size(::FODData)` returns the size of the data.
- `get_range(::FODData)` returns the range of the data in the real world coordinates.
- `is_normalized(::FODData)` return whether the data is normalized.

## Constructors

In the following, `data` has shape `nx x ny x nz x nt`.

- `FODData(data::AbstractArray{𝒯, 4}, transform::Transform, normalize_it::Bool; file_name = "None")`.
- `FODData(data::AbstractArray{𝒯, 4}, S, T, normalize_it::Bool; file_name = "None")`.
- `FODData(data::AbstractArray{𝒯, 4}, normalize_it::Bool)` uses a trivial transform.
- `FODData(file_name::String; normalize_it::Bool = true, k...)` the transform is extracted from the nii file.
"""
struct FODData{𝒯, 𝒯d, 𝒯s, 𝒯t, 𝒯b, 𝒯i}
    "filename from which the (fod) data is read."
    filename::String
    "field which contains the FOD data."
    data::𝒯d
    "max l coordinate in of spherical harmonics."
    lmax::Int
    "transform associated with data, see `Transform`."
    transform::Transform{𝒯s, 𝒯t}
    "Are the data normalized? In this case `fod[i,j,l,1] ∈ {0,1}`."
    normalized::Bool
    "Basis for fODF. For example `SphericalHarmonics()`"
    basis::𝒯b
    "Interpolation of fODF"
    interpolation::𝒯i
end

"""
$(TYPEDSIGNATURES)

max `l` coordinate in of spherical harmonics.
"""
@inline get_lmax(foddata::FODData) = foddata.lmax
Base.size(foddata::FODData) = size(foddata.data)
_get_array(x::AbstractArray) = x
_get_array(x::NIfTI.NIVolume) = x.raw
_get_array(x::FODData) = _get_array(x.data)
_my_typeof(x) = typeof(x)
_my_typeof(x::NIfTI.NIVolume) = typeof(x.raw)
is_normalized(foddata) = foddata.normalized
get_basis(foddata) = foddata.basis

"""
$(TYPEDSIGNATURES)

Returns the range of the data in the real world coordinates
"""
function get_range(foddata::FODData)
    nx, ny, nz, nt = size(foddata)
    lx, ly, lz = transform(foddata, SA.SVector(1, 1, 1))
    rx, ry, rz = transform(foddata, SA.SVector(nx, ny, nz))
    return sort(LinRange(lx, rx, nx)), 
           sort(LinRange(ly, ry, ny)),
           sort(LinRange(lz, rz, nz))
end

"""
$(TYPEDSIGNATURES)

Constructor for `FODData` based on Array data and transform.

## Arguments
- `data::AbstractArray{𝒯, 4}`.
- `transform::Transform` affine mapping for the coordinate transform.
- `normalize_it (= true)` the raw spherical harmonics are scaled so that the zero spherical harmonic coefficient is one (or zero).

## Keyword arguments
- `file_name (= "None")` path to data.
"""
function FODData(data::AbstractArray{𝒯, 4}, 
                    transform::Transform{𝒯s, 𝒯t},
                    normalize_it::Bool;
                    file_name = "None",
                    basis::𝒯b = SphericalHarmonics(),
                    interpolation::𝒯i = NearestNeighbor()) where {𝒯, 𝒯s, 𝒯t, 𝒯b, 𝒯i}
    lmax = get_lmax_from_fod_length(size(_get_array(data), 4))
    FODData{_my_typeof(data), typeof(data), 𝒯s, 𝒯t, 𝒯b, 𝒯i}(file_name, data, lmax, transform, normalize_it, basis, interpolation)
end

"""
$(TYPEDSIGNATURES)

Constructor for `FODData` based on Array data and transform.

## Arguments
- `data::AbstractArray`. Should be 4d array for spherical harmonics basis.
- `S` linear mapping for the coordinate transform. It will be passed to `Transform(S, T)`.
- `T` translation for the coordinate transform. It will be passed to `Transform(S, T)`.
- `normalize_it (= true)` the raw spherical harmonics are scaled so that the zero spherical harmonics coefficient is one (or zero).

## Keyword arguments
- `file_name (= "None")` path to data.
"""
function FODData(data::AbstractArray{𝒯, 4}, 
                S::𝒯s, 
                T::𝒯t, 
                normalize_it::Bool;
                file_name = "None",
                basis = SphericalHarmonics{true}(),
                interpolation = NearestNeighbor()) where {𝒯, 𝒯s, 𝒯t}
    FODData(data, Transform(S, T), normalize_it; file_name, basis, interpolation)
end

"""
$(TYPEDSIGNATURES)

Constructor for `FODData` based on Array data and trivial transform.
"""
FODData(data::AbstractArray{𝒯, 4}, normalize_it::Bool; basis = SphericalHarmonics{true}(), interpolation = NearestNeighbor()) where {𝒯} = 
    FODData(data, 
            SA.SMatrix{4, 4, 𝒯}(I(4)), 
            SA.SVector{3, 𝒯}(zeros(𝒯, 3)), 
            normalize_it; 
            basis,
            interpolation)

@inline transform(ni::FODData, x) = transform(ni.transform, x)
@inline transform_inv(ni::FODData, x) = transform_inv(ni.transform, x)

"""
$(TYPEDSIGNATURES)

Constructor for `FODData` based on a NII file.
Read a `.nii.gz` or a `.nii` file passed as a `String`.

You can display more information using 

```
show(stdout, ni; full = true)
```

## Arguments
- `file_name` path to a nii file.

## Keyword arguments
- `normalize_it (= true)` the raw spherical harmonics are scaled so that the zero spherical harmonic coefficient is one (or zero).
- `basis [= SphericalHarmonics()]`
- the additional keyword arguments are passed to `NIfTI.niread`.

## Output

It returns a `FODData` struct.
"""
function FODData(file_name::String; 
                normalize_it::Bool = true, 
                basis = SphericalHarmonics{true}(),
                interpolation = NearestNeighbor(),
                k...) 
    data = niread(file_name; k...)
    # we normalize the ODF to have mass one
    if basis isa SphericalHarmonics && 
        ~all(x-> x >= 0, data.raw[:,:,:,1]) 
        @warn "Some zero SH coefficients are negative!\nPutting them to zero"
    end
    if normalize_it
        _normalize_data!(data)
    end
    @debug size(data) data.header
    A = NIfTI.getaffine(data.header)
    S = SA.@SMatrix [A[i, j] for i = 1:4, j = 1:4]
    T = SA.@SVector [A[i, end] for i = 1:3]
    return FODData(data, S, T, normalize_it; file_name, basis, interpolation)
end

"""
$(TYPEDSIGNATURES)

Normalize the data so that the first coefficient belongs to {0, 1}.
"""
function _normalize_data!(data::AbstractArray{T, N}) where {T,N}
    if N < 4; return; end
    nx,ny,nz, = size(data.raw)
    Threads.@threads for k=1:nz
        for j=1:ny
            for i=1:nx
                α = data.raw[i,j,k,1]
                if α > 0
                    data.raw[i,j,k,:] ./= α
                end
            end
        end
    end
end
###########################################################################
"""
$(TYPEDSIGNATURES)
"""
function Base.show(io::IO, foddata::FODData{T, Tp}; full::Bool = false, prefix = "") where {T, Tp} 
    printstyled(prefix, Tp, "\n", bold = true, color = :cyan)
    println(prefix * " ├─ File name     = ", foddata.filename)
    println(prefix * " ├─ Dimensions    = ", size(foddata.data))
    println(prefix * " ├─ basis         = ", get_basis(foddata))
    println(prefix * " ├─ interpolation = ", foddata.interpolation)
    if get_basis(foddata) isa SphericalHarmonics
        println(prefix * " ├─ lmax (SH)     = ", foddata.lmax)
        println(prefix * " ├─ normalized    = ", foddata.normalized)
    end
    if foddata.data isa NIfTI.NIVolume
        println(prefix * " ├─ Voxel size  = ", foddata.data.header.pixdim[1:4])
        println(prefix * " ├─ Orientation = ", NIfTI.orientation(foddata.data))
    end
    println(prefix * " └─ Transform (s_row) = ⋯")
    if full
        foddata.transform.S |> display
    end
end