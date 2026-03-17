function _correct_normals!(pts, faces)
    @assert size(faces)[2] == 3
    for i in axes(faces, 1)  
        @views i1, i2, i3 = faces[i, :]
        v1 = SVector(pts[i1]) - SVector(pts[i2])
        v2 = SVector(pts[i3]) - SVector(pts[i2])
        if dot(cross(v1, v2), pts[i1]) < 0 
            faces[i, 1] = i3
            faces[i, 3] = i1
        end
    end
end

function _get_sphere_fibonacci(N, 𝒯::DataType = Float64)
    angles = Tractography.fibonacci_sampling(N, 𝒯)
    directions = [spherical_to_euclidean(d...) for d in angles]
    hull = Quickhull.quickhull(directions)

    faces = mapreduce(x-> [x.data...], hcat, Quickhull.facets(hull))'
    _correct_normals!(directions, faces)

    pts_h = Matrix(reduce(hcat, [[p...] for p in directions])')
    return pts_h, faces, angles
end
####################################################################################################
"""
$(SIGNATURES)

Add local frame to the scene `sc`.
"""
function add_frame!(sc; x0 = zeros(3), r = 1, k...)
    lines!(sc, Point3.([x0, x0 .+ [r, 0, 0]]); color = :red, k...)
    lines!(sc, Point3.([x0, x0 .+ [0, r, 0]]); color = :green, k...)
    lines!(sc, Point3.([x0, x0 .+ [0, 0, r]]); color = :blue, k...)
end

"""
$(SIGNATURES)

Plot the streamlines with lines colored by local orientation.

The optional parameters are passed to `lines!`.
"""
@views function plot_streamlines!(ax, streamlines::AbstractArray{𝒯, 3}; k...) where {𝒯}
    _colors = Makie.RGB{𝒯}[]
    _lines = Point{3, 𝒯}[]
    @time  "SL" for nm in axes(streamlines, 3)
        for i in axes(streamlines, 2)
            push!(_lines, Point3(streamlines[1:3, i, nm]...))
            if i == 1
                x, y, z = 0, 0, 0
            else
                x = streamlines[1, i, nm] - streamlines[1, i-1, nm]
                y = streamlines[2, i, nm] - streamlines[2, i-1, nm]
                z = streamlines[3, i, nm] - streamlines[3, i-1, nm]
                nrm = sqrt(x^2 + y^2 + z^2)
                if nrm == 0
                    x, y, z = 0, 0, 0
                else
                    x /= nrm; y /= nrm; z /= nrm
                end
            end
            push!(_colors, Makie.RGB(abs(x), 
                                     abs(y), 
                                     abs(z)
                                    ) 
                )
            if (x^2 + y^2 + z^2 == 0) && i > 1
                break
            end
        end
        # this breaks the line
        push!(_lines, Point{3, 𝒯}(NaN))
        push!(_colors, Makie.RGB{𝒯}(NaN))
    end

    lines!(ax, 
            _lines;
            color = _colors,
            k...
            )
    return ax
end
####################################################################################################
function plot_fod(model::TMC; k...)
    f = Figure(backgroundcolor = :black)
    lscene = LScene(f[1, 1], scenekw = (lights = [
                            AmbientLight(RGBf(1, 1, 1)), 
                            DirectionalLight(RGBf(1, 1, 1), Vec3f(-1, 0, 1)),
                            DirectionalLight(RGBf(1, 1, 1), Vec3f(-1, 0, -1))
                        ],)
                )
    plot_fod!(lscene, model; k...)
    f, lscene
end

"""
$(SIGNATURES)

Plot the the ODF with glyphs.

## See also
- `plot_fod(model; kwargs...)` for an out-of-place method with same parameters.

## Arguments
- `ax` GLMakie scene
- `model::TMC`

## Optional arguments
- `I, J, K`: range for displaying the fODFs. Some of them can be a single element, like `K = 40:40`.
- `radius`: radius of the glyph.
- `st = 4`: stride, only show one over `st` glyph in each direction.
"""
function plot_fod!(ax, model::TMC{𝒯} ; 
                    n_sphere = 10,
                    radius = 0.1,
                    st = 4,
                    I = nothing,
                    J = nothing,
                    K = nothing,
                    c0min = 0.1) where 𝒯

    # Makie is happier with Vector{Point3} and Vector{GLTriangleFace}
    # it will convert to this anyway
    # TODO use those structures
    ni = getdata(model)
    nx, ny, nz, nt = size(ni)
    odf = zeros(Float32, nt)
    cache = _init((@set model.evaluation_algo = Tractography.PlottingSH()), Probabilistic(); n_sphere)
    Yₗₘ = Float32.(cache.Yₗₘ)
    F = zeros(Float32, length(cache.angles))
    radius = Float32(radius)

    # this is to ensure better glyph plotting
    pts, faces, angles = _get_sphere_fibonacci(n_sphere, Float32)
    n_faces0 = length(axes(faces, 1))
    n_pts0 = length(axes(pts, 1))

    @assert pts isa Matrix{Float32}

    _colors = [Makie.RGB(abs(sin(θ) * cos(ϕ)), 
                         abs(sin(θ) * sin(ϕ)), 
                         abs(cos(θ))) for (θ, ϕ) in angles]

    Is = isnothing(I) ? (1:st:nx) : I
    Js = isnothing(J) ? (1:st:ny) : J
    Ks = isnothing(K) ? (1:st:nz) : K
    @info "Dimension" size(Is) size(Js) size(Ks) Is Js Ks c0min

    n_glygths = sum(ni.data[Is, Js, Ks, 1] .> c0min)
    all_pts   = Matrix{Float32}(undef, n_glygths * n_pts0, 3) 
    all_faces = Matrix{UInt32}(undef, n_faces0 * n_glygths, 3)

    n_elements = 0
    nfaces = 1
    l_pts::Int = 0
    @time "LOOP" for i in Is
        x0 = i
        for j in Js
            y0 = j
            for k in Ks
                z0 = k
                c₀₀ = ni.data[i, j, k, 1]
                if c₀₀ > c0min
                    odf .= @view ni.data[i, j, k, :]
                    mul!(F, Yₗₘ, odf)
                    X0 = transform(ni, SVector(x0-1, y0-1, z0-1))

                    @inbounds for idx in 1:n_pts0
                        r = model.mollifier(F[idx]) * radius / c₀₀
                        all_pts[l_pts + idx, 1] = pts[idx, 1] * r + X0[1]
                        all_pts[l_pts + idx, 2] = pts[idx, 2] * r + X0[2]
                        all_pts[l_pts + idx, 3] = pts[idx, 3] * r + X0[3]
                    end

                    all_faces[nfaces:(nfaces + n_faces0 - 1), :] .= faces .+ l_pts
                    l_pts += n_pts0
                    nfaces += n_faces0
                    n_elements += 1
                end
            end
        end
    end
    @info "Glyph number = " n_elements 
    @info "Glypth components = " length(vec(F))
    @info "Arrays" size(all_pts) size(all_faces)

    mesh!(
        ax, 
        all_pts,
        all_faces;
        # strokewidth = 1,
        shading = FastShading,
        color = repeat(_colors, n_elements)
    )
    ax
end

"""
$(SIGNATURES)

Plot a slice of the data.

## See also
- `plot_slice!`

## Arguments
- `model::TMC`

## Optional arguments
- `odf::Bool` display the ODF with glyphs.
- `slice::Bool` display the brain slice from mean ODF value.
- `interpolate::Bool` interpolate the brain slice (reduces pixelization).
- `alpha` alpha value for brain slice.
- `I,J,K` range for displaying the ODF. One of them should be a single element, like `K = 40:40`.
- `st::Int = 2`
- `c0min` minimum mean ODF value for plotting the glyph.
- `n_theta::Int` number of points for showing the glyphs.
- `radius = 0.1` radius of the glyphs.
"""
function plot_slice(model; k...)
    f = Figure(backgroundcolor = :black, size = (1000, 1200))
    sc = LScene(f[1, 1], scenekw = (lights = [
                            AmbientLight(RGBf(1, 1, 1)), 
                            DirectionalLight(RGBf(1, 1, 1), Vec3f(-1, 0, 1)),
                            DirectionalLight(RGBf(1, 1, 1), Vec3f(-1, 0, -1))
                        ],)
                )
    plot_slice!(sc, model; k...)
    f, sc
end

function plot_slice!(ax, model; 
                    st::Int = 1, 
                    odf::Bool = true,
                    slice::Bool = true,
                    I = nothing, 
                    J = nothing, 
                    K = nothing, 
                    c0min = 0.13, 
                    n_sphere::Int = 10, 
                    radius = 0.1,
                    alpha = 1.,
                    interpolate::Bool = true)
    ni = getdata(model)
    nx, ny, nz, nt = size(ni.data)

    Is = isnothing(I) ? (1:nx) : I
    Js = isnothing(J) ? (1:ny) : J
    Ks = isnothing(K) ? (1:nz) : K

    Io = isnothing(I) ? (1:st:nx) : I
    Jo = isnothing(J) ? (1:st:ny) : J
    Ko = isnothing(K) ? (1:st:nz) : K

    # do we plot the glyphs?
    if odf
        @time_debug "ODF" plot_fod!(ax, model; n_sphere, radius, 
                    c0min,
                    I = Io,
                    J = Jo,
                    K = Ko)
    end

    Ih = length(Is) == 1 ? Is[1] : Is
    Jh = length(Js) == 1 ? Js[1] : Js
    Kh = length(Ks) == 1 ? Ks[1] : Ks

    plane = :xy; lplane = Ks[1]; vox_frame = [0, 0, Ks[1]]
    if length(Ih) == 1
        plane = :yz
        axe1 = [transform(ni,  SVector(Ih[1], j,     Kh[1]))[2] for j in Jh]
        axe2 = [transform(ni,  SVector(Ih[1], Jh[1], k)    )[3] for k in Kh]
        lplane = transform(ni, SVector(Ih[1], Jh[1], Kh[1]))[1]
        vox_frame = [Ih[1], 0, 0]
        native_frame = [lplane, 0, 0]
        v_frame = [lplane, axe1[end÷2], axe2[end÷2]]
    elseif length(Jh) == 1
        plane = :xz
        axe1 = [transform(ni,  SVector(i,     Jh[1], Kh[1]))[1] for i in Ih]
        axe2 = [transform(ni,  SVector(Ih[1], Jh[1], k)    )[3] for k in Kh]
        lplane = transform(ni, SVector(Ih[1], Jh[1], Kh[1]))[2]
        vox_frame = [0, Jh[1], 0]
        native_frame = [0, lplane, 0]
        v_frame = [axe1[end÷2], lplane, axe2[end÷2]]
    elseif length(Kh) == 1
        plane = :xy
        axe1 = [transform(ni,  SVector(i,     Jh[1], Kh[1]))[1] for i in Ih]
        axe2 = [transform(ni,  SVector(Ih[1], j,     Kh[1]))[2] for j in Jh]
        lplane = transform(ni, SVector(Ih[1], Jh[1], Kh[1]))[3]
        vox_frame = [0, 0, Kh[1]]
        native_frame = [0, 0, lplane]
        v_frame = [axe1[end÷2], axe2[end÷2], lplane]
    end

    if slice
        hm = heatmap!(ax, 
                      axe1,
                      axe2,
                      ni.data[Ih, Jh, Kh, 1]; 
                      colormap = :grays, 
                      transformation = (plane, lplane), 
                      interpolate, 
                      alpha)
        # translate!(hm, Vec3f(Is[1], Js[1], Ks[1]+1))
    end

    add_frame!(ax; r = 10, x0 = v_frame, linewidth = 3)

    # add slice information
    Label(ax.parent, "   voxel(s): " * replace("$vox_frame", "0" => ":") * "\nposition: $(native_frame) (mm)", 
            tellheight = false, 
            tellwidth = false, 
            halign = :left, 
            valign = :top, 
            color = :red)
end