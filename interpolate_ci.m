function [ci_low, ci_high] = interpolate_ci(prange, norm_profile, threshold)
    ci_low = NaN; ci_high = NaN;
    try
        for k = 1:length(prange)-1
            if norm_profile(k) < threshold && norm_profile(k+1) >= threshold
                ci_low = (threshold*(prange(k+1)-prange(k)) + norm_profile(k+1)*prange(k) - norm_profile(k)*prange(k+1)) / (norm_profile(k+1) - norm_profile(k));
            end
            if norm_profile(k) >= threshold && norm_profile(k+1) < threshold
                ci_high = (threshold*(prange(k+1)-prange(k)) + norm_profile(k+1)*prange(k) - norm_profile(k)*prange(k+1)) / (norm_profile(k+1) - norm_profile(k));
            end
        end
    catch, end
end