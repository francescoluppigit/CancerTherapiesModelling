function [t_sub, Ndata_sub] = subsample_data_full_random(t, Ndata, n_to_remove)
    total_points = length(t);
    MIN_POINTS_FOR_FIT = 5; 
    points_to_keep = total_points - n_to_remove;

    if n_to_remove < 0
        t_sub = t;
        Ndata_sub = Ndata;
        return;
    end
    
    if points_to_keep < MIN_POINTS_FOR_FIT
        % ERROR: not enough points
        % fprintf('WARNING: Cannot remove %d points... Returning empty.\n', n_to_remove);
        t_sub = []; 
        Ndata_sub = []; 
        return;
    end

    % --- Random Subsampling ---
    idx_to_keep = randperm(total_points, points_to_keep);
    idx_to_keep = sort(idx_to_keep);
    t_sub = t(idx_to_keep);
    Ndata_sub = Ndata(idx_to_keep);
end