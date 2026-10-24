using Test, Tractography
const TG = Tractography

function read_tck(filename)
    bytes = read(filename)
    magic = codeunits("mrtrix tracks")
    @test bytes[1:length(magic)] == magic

    marker = findfirst(codeunits("\nEND\n"), bytes)
    stop = last(marker)
    header = String(bytes[1:stop])

    count_match = match(r"count:\s*(\d+)", header)
    offset_match = match(r"file:\s*\.\s*(\d+)", header)
    count = parse(Int, count_match.captures[1])
    offset = parse(Int, offset_match.captures[1])

    data_offset = stop + 1
    raw = reinterpret(UInt32, bytes[data_offset:end])
    points = reinterpret(Float32, ltoh.(raw))
    return header, count, points, offset, data_offset
end

@testset "save_streamlines" begin
    tractogram = Float32.(reshape(1:36, 3, 4, 3))
    tracto_length = UInt32[4, 2, 3]

    mktempdir() do dir
        filename = joinpath(dir, "streamlines.tck")
        TG.save_streamlines_julia(filename, tractogram, tracto_length)

        header, count, points, offset, data_offset = read_tck(filename)
        @test startswith(header, "mrtrix tracks\n")
        @test occursin("count: 0000000002", header)
        @test occursin("datatype: Float32LE", header)
        @test offset == data_offset - 1
        @test count == 2

        expected = Float32[]
        for k in (1, 3)
            for j in 1:tracto_length[k]
                push!(expected, tractogram[1, j, k], tractogram[2, j, k], tractogram[3, j, k])
            end
            push!(expected, NaN32, NaN32, NaN32)
        end
        push!(expected, Inf32, Inf32, Inf32)

        @test length(points) == length(expected)
        @test all(isequal.(points, expected))
    end

    mktempdir() do dir
        filename = joinpath(dir, "empty.tck")
        TG.save_streamlines_julia(filename, tractogram, [0, 1, 2])

        header, count, points, offset, data_offset = read_tck(filename)
        @test occursin("count: 0000000000", header)
        @test offset == data_offset - 1
        @test count == 0
        @test all(isequal.(points, [Inf32, Inf32, Inf32]))
    end

    mktempdir() do dir
        filename = joinpath(dir, "float64.tck")
        TG.save_streamlines_julia(filename, Float64.(tractogram), [4, 4, 4])

        _, count, points, _, _ = read_tck(filename)
        @test count == 3
        @test points[1:12] == vec(tractogram[:, 1:4, 1])
        @test all(isnan, points[13:15])
    end

    mktempdir() do dir
        filename = joinpath(dir, "invalid.tck")
        @test_throws ErrorException TG.save_streamlines_julia(filename, rand(Float32, 2, 4, 1), [4])
        @test_throws ErrorException TG.save_streamlines_julia(filename, rand(Float32, 3, 4, 1), [5])
        @test_throws ErrorException TG.save_streamlines_julia(filename, rand(Float32, 3, 4, 2), [4])
    end
end
