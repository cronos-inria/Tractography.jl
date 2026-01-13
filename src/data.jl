"""
$(TYPEDEF)

Structure to hold the affine transform from the real world to voxel coordinates.

# Fields
$(TYPEDFIELDS)

# Methods
- see `transform(tf::Transform, x) `
"""
struct Transform{𝒯s, 𝒯t}
    "Forward transform."
    S::𝒯s
    "Inverse transform."
    Sinv::𝒯s
    "Translation."
    T::𝒯t
end
@inline transform(tf::Transform, x) = tf.S * x # for plotting
@inline transform(tf::Transform, x::CartesianIndex{3}) = transform(tf, SA.SVector(Tuple(x)..., 1)) # for plotting
@inline transform(tf::Transform, x::SA.SVector{3}) = transform(tf, SA.SVector(x..., 1)) # for plotting
@inline transform_inv(tf::Transform, x) = tf.Sinv * x

"""
$(TYPEDEF)

Structure to hold FOD data.

# Fields
$(TYPEDFIELDS)

# Methods
- `get_lmax(::FODData)` returns the max `l` coordinate in of spherical harmonics.
- `size(::FODData)` returns the size of the data.
- `get_range(::FODData)` returns the range of the data in the real world coordinates.
"""
struct FODData{𝒯, 𝒯d, 𝒯s, 𝒯t}
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
end

"""
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
- `data::AbstractArray{𝒯, 4}`
"""
function FODData(file_name, data::AbstractArray{𝒯, 4}, S::𝒯s, T::𝒯t, normalize_it::Bool) where {𝒯, 𝒯s, 𝒯t}
    Sinv = pinv(S)
    lmax = get_lmax_from_fod_length(size(_get_array(data), 4))
    FODData{_my_typeof(data), typeof(data), 𝒯s, 𝒯t}(file_name, data, lmax, Transform(S, Sinv, T), normalize_it)
end
@inline transform(ni::FODData, x) = transform(ni.transform, x)
@inline transform_inv(ni::FODData, x) = transform_inv(ni.transform, x)

"""
$(TYPEDSIGNATURES)

Constructor for `FODData` based on NII file.
Read a `.nii.gz` or a `.nii` file passed as a `String`.

The raw spherical harmonics are scaled so that the zero spherical harmonic coefficient is one (or zero).

You can display more information using 

```
show(stdout, ni; full = true)
```

## Output

It returns a `FODData` struct.
"""
function FODData(file::String; normalize_it::Bool = true, k...) 
    data = niread(file; k...)
    # we normalize the ODF to have mass one
    if ~all(x-> x >= 0, data.raw[:,:,:,1]) 
        @warn "Some zero SH coefficients are negative!\nPutting them to zero"
    end
    if normalize_it
        _normalize_sph_data!(data)
    end
    @debug size(data) data.header
    A = NIfTI.getaffine(data.header)
    S = SA.@SMatrix [A[i, j] for i = 1:4, j = 1:4]
    T = SA.@SVector [A[i, end] for i = 1:3]
    return FODData(file, data, S, T, normalize_it)
end

function _normalize_sph_data!(data)
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
function Base.show(io::IO, ni::FODData{T, Tp}; full::Bool = false, prefix = "") where {T, Tp} 
    printstyled(prefix, Tp, "\n", bold = true, color = :cyan)
    println(prefix * " ├─ File name   = ", ni.filename)
    println(prefix * " ├─ lmax (SH)   = ", ni.lmax)
    println(prefix * " ├─ Dimensions  = ", size(ni.data))
    println(prefix * " ├─ normalized  = ", ni.normalized)
    if ni.data isa NIfTI.NIVolume
        println(prefix * " ├─ Voxel size  = ", ni.data.header.pixdim[1:4])
        println(prefix * " ├─ Orientation = ", NIfTI.orientation(ni.data))
    end
    println(prefix * " └─ Transform (s_row) = ⋯")
    if full
        ni.transform.S |> display
    end
end