####################################################################################################
const _TCK_MAGIC = "mrtrix tracks"
const _TCK_FIBER_DELIMITER = htol.([
    reinterpret(UInt32, NaN32),
    reinterpret(UInt32, NaN32),
    reinterpret(UInt32, NaN32),
])
const _TCK_EOF_DELIMITER = htol.([
    reinterpret(UInt32, Inf32),
    reinterpret(UInt32, Inf32),
    reinterpret(UInt32, Inf32),
])

@inline _tck_float32(x) = htol(reinterpret(UInt32, Float32(x)))

function _tck_header(n_fibres::Integer)
    prefix = string(_TCK_MAGIC, '\n',
                    "count: ", lpad(n_fibres, 10, '0'), '\n',
                    "datatype: Float32LE")
    offset = 0
    for _ in 1:8
        header = string(prefix, "\nfile: . ", offset, "\nEND\n")
        new_offset = ncodeunits(header)
        new_offset == offset && return header
        offset = new_offset
    end
    return string(prefix, "\nfile: . ", offset, "\nEND\n")
end
####################################################################################################
"""
$(SIGNATURES)

Save a tractogram in `filename` using the `tck` format.

The file is written natively in Julia, without requiring Python or `nibabel`.
Coordinates are stored as little-endian `Float32`.

# Arguments
- `filename::String` file name.
- `tractogram` is an abstract array of dimension `3 × n_points × n_fibres`.
- `tracto_length` gives the number of points of each fibre. Fibres with two
  points or less are not saved.
"""
function save_streamlines_julia(filename::String,
                          tractogram::AbstractArray{T, 3},
                          tracto_length) where {T}
    size(tractogram, 1) == 3 ||
        error("The streamlines must have size 3 x ns x n_nmc")

    n_fibres = size(tractogram, 3)
    lengths = Int.(collect(tracto_length))
    length(lengths) >= n_fibres ||
        error("tracto_length must have at least $(n_fibres) entries")

    n_points_max = size(tractogram, 2)
    n_saved = 0
    for k in 1:n_fibres
        n_points = lengths[k]
        if n_points > 2
            n_points <= n_points_max ||
                error("tracto_length[$k] = $n_points exceeds the number of allocated points $n_points_max")
            n_saved += 1
        end
    end

    @info "Saving in file $filename in Julia native..."

    open(filename, "w") do io
        write(io, _tck_header(n_saved))
        buffer = Vector{UInt32}(undef, 3 * n_points_max)
        for k in 1:n_fibres
            n_points = lengths[k]
            n_points > 2 || continue

            index = 1
            @inbounds for j in 1:n_points
                buffer[index]     = _tck_float32(tractogram[1, j, k])
                buffer[index + 1] = _tck_float32(tractogram[2, j, k])
                buffer[index + 2] = _tck_float32(tractogram[3, j, k])
                index += 3
            end

            GC.@preserve buffer unsafe_write(io, Ptr{UInt8}(pointer(buffer)), UInt(12 * n_points))
            write(io, _TCK_FIBER_DELIMITER)
        end
        write(io, _TCK_EOF_DELIMITER)
    end

    return nothing
end
