module UMPABackend

export umpa_match

function _as_frame_list(frames)
    if frames isa AbstractArray && ndims(frames) == 3
        return [view(frames, i, :, :) for i in axes(frames, 1)]
    end
    return frames
end

function _accumulate(frames, masks, positions, full_shape)
    sums = zeros(Float64, full_shape...)
    sums_sq = zeros(Float64, full_shape...)
    counts = zeros(Float64, full_shape...)

    for idx in eachindex(frames)
        frame = Array(frames[idx])
        pos = positions[idx, :]
        # Positions from Python are 0-based row/col values (pos[0], pos[1]);
        # in Julia these arrive as pos[1]/pos[2], so add 1 to map to 1-based indices.
        y0 = Int(round(pos[1])) + 1
        x0 = Int(round(pos[2])) + 1
        h, w = size(frame)
        y1 = y0 + h - 1
        x1 = x0 + w - 1

        if masks === nothing
            @views sums[y0:y1, x0:x1] .+= frame
            @views sums_sq[y0:y1, x0:x1] .+= frame .^ 2
            @views counts[y0:y1, x0:x1] .+= 1
        else
            mask = Array(masks[idx])
            @views sums[y0:y1, x0:x1] .+= frame .* mask
            @views sums_sq[y0:y1, x0:x1] .+= (frame .^ 2) .* mask
            @views counts[y0:y1, x0:x1] .+= mask
        end
    end

    safe_counts = max.(counts, 1)
    mean = sums ./ safe_counts
    var = sums_sq ./ safe_counts .- mean .^ 2
    return mean, var, counts
end

function _crop(array, padding)
    if padding <= 0
        return array
    end
    if size(array, 1) <= padding * 2 || size(array, 2) <= padding * 2
        return zeros(Float64, 0, 0)
    end
    return array[(padding + 1):(end - padding), (padding + 1):(end - padding)]
end

function umpa_match(sample, ref, mask, positions, full_shape, padding; include_df=true)
    sample_frames = _as_frame_list(sample)
    ref_frames = _as_frame_list(ref)
    mask_frames = mask === nothing ? nothing : _as_frame_list(mask)

    mean_s, var_s, coverage = _accumulate(sample_frames, mask_frames, positions, full_shape)
    mean_r, var_r, _ = _accumulate(ref_frames, mask_frames, positions, full_shape)

    mean_s = _crop(mean_s, padding)
    mean_r = _crop(mean_r, padding)
    var_s = _crop(var_s, padding)
    var_r = _crop(var_r, padding)
    coverage = _crop(coverage, padding)

    scale = isempty(mean_r) ? 1.0 : max(maximum(abs.(mean_r)), 1.0)
    denom = max.(mean_r, eps(Float64) * scale)
    T = mean_s ./ denom
    f = (mean_s .- mean_r) .^ 2
    dx = zeros(Float64, size(T))
    dy = zeros(Float64, size(T))
    var_scale = isempty(var_r) ? 1.0 : max(maximum(var_r), 1.0)
    df = include_df ? sqrt.(max.(var_s, 0.0) ./ max.(var_r, eps(Float64) * var_scale)) : nothing

    return Dict(
        "T" => T,
        "dx" => dx,
        "dy" => dy,
        "df" => df,
        "f" => f,
        "coverage" => coverage,
        "Im" => mean_r,
    )
end

end
