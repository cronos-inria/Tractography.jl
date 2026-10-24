module PythonExt
    using DocStringExtensions, PythonCall
    import Tractography: save_streamlines

    """
    $(SIGNATURES)

    Save tractogram in `filename` in a `tck` format using the Python package nibabel. You should probably use a filename like `streamlines-julia.tck`.

    # Arguments
    - `filename::String` file name.
    - `tractogram` can either be an abstract array of dimension `3 x n_length_fiber x n_fibres`.

    # Notes
    The current implementation is based on NiBabel.
    """
    function save_streamlines(filename::String, 
                                tractogram::AbstractArray{T, 3}, 
                                tracto_length) where {T}
        if size(tractogram, 1) != 3 
            error("The streamlines must have size 3 x ns x n_nmc")
        end
        nib = pyimport("nibabel")
        np = pyimport("numpy")
        streamlines = [( tractogram[:, 1:tracto_length[k], k])' for k in axes(tractogram, 3) if tracto_length[k] > 2]
        @info "Saving in file $filename by calling nibabel"
        nib.streamlines.TckFile(nib.streamlines.Tractogram(streamlines, affine_to_rasmm = np.eye(4))).save(filename)
    end
end