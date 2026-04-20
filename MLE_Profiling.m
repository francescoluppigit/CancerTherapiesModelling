%%% 1. CONFIGURATION & SETUP %%%
clear; clc; close all;

% Dataset definitions
data_folder1 = 'Dati1\stomatitis\A549'; data_folder2 = 'Dati1\relaxin\C33A';
data_folder3 = 'Dati1\immune'; data_folder4 = 'Dati1\collagenase';
data_folder5 = 'Dati1\hyaluronidase'; data_folder6 = 'Dati1\measles';
data_folder7 = 'Dati1\relaxin\rA549'; data_folder8 = 'Dati1\relaxin\U343';
data_folder9 = 'Dati1\reovirus'; data_folder10 = 'Dati1\stroma\sA549 Tcell';
data_folder11 = 'Dati1\stroma\sA549';

data_folder12 = 'Dati2\chimeric\RIG\injected'; data_folder13 = 'Dati2\chimeric\RIG\uninjected';
data_folder14 = 'Dati2\chimeric\WT\injected'; data_folder15 = 'Dati2\chimeric\WT\uninjected';
data_folder16 = 'Dati2\amplicon'; data_folder17 = 'Dati2\killer';
data_folder18 = 'Dati2\glycolidase'; data_folder19 = 'Dati2\combined';
data_folder20 = 'Dati2\measles'; data_folder21 = 'Dati2\vaccinia';

% Active settings
data_folder = data_folder1; 
model_type  = 'Gompertz'; 
t_start     = 11; 

% Profiling settings
profile_npts = 100;  
profile_TH   = -1.92; % 95% CI threshold (Chi-Square)

options = optimoptions('fmincon', 'Display', 'off', ...
    'MaxFunctionEvaluations', 5000, 'OptimalityTolerance', 1e-10);

% Directories and logging
[folder_path, folder_name_base, ~] = fileparts(data_folder);
if isempty(folder_name_base), [~, folder_name_base, ~] = fileparts(folder_path); end

save_dir = fullfile('img', folder_name_base);
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

log_filename = fullfile(save_dir, sprintf('Log_Profiling_%s_%s_%s.txt', model_type, folder_name_base, datestr(now, 'yyyymmdd_HHMM')));
diary(log_filename);

fprintf('Profile Likelihood Analysis\nDataset: %s\nModel: %s\nTime Filter (t >= %d)\nDate: %s\n\n', ...
    data_folder, model_type, t_start, datetime("now"));

%%% 2. DATA IMPORT & PARAMETER MAPPING %%%
[data_paths, curve, L] = load_data_set(data_folder); 

% Adjusted logic for targeted parameter profiling and specific grid layouts
switch lower(model_type)
    case {'logistic', 'gompertz', 'logisticlogn', 'gompertzlogn'}
        active_idx = [1, 2, 3]; % lambda, K, N0
        plot_param_names = {'r', 'K', 'C(0)'};
        n_rows = 1; n_cols = 3;
        fig_size = [100, 200, 1400, 450];
    case {'richards', 'richardslogn'}
        active_idx = [1, 5, 6, 2]; % lambda, beta, lambda*beta, K
        plot_param_names = {'r', '\beta', 'r \cdot \beta', 'K'};
        n_rows = 1; n_cols = 4;
        fig_size = [50, 200, 1400, 450];
end
num_active_params = length(active_idx);
plot_fs = 20;

%%% 3. MAIN LOOP %%%
for i = 1:L
    % Import raw data for initial estimates, then filter for fitting
    [t_raw, Ndata_raw] = import_data(data_paths{i});
    
    idx_valid = (t_raw >= t_start);
    t = t_raw(idx_valid);
    Ndata = Ndata_raw(idx_valid);
    n_p = length(t);
    
    if n_p < 6
        fprintf('Dataset %s skipped (insufficient valid points).\n', curve{i});
        continue;
    end
    
    % Use Ndata_raw to define realistic boundaries and initial values
    [l0, K0, N00, s0, b0, nonlcon, lb, ub] = initialise_param(model_type, Ndata_raw);
    
    try
        % Fit strictly on the filtered interval
        [mle, nLL] = fit_growth_model_local(model_type, t, Ndata, [l0, K0, N00, s0, b0], lb, ub, nonlcon, options);
    catch ME
        fprintf('FIT FAILED | %s: %s\n', curve{i}, ME.message);
        continue;
    end
    
    [k_val, ssr, aic, Nmle, res, mape] = evaluate_fit(model_type, mle, t, Ndata, nLL);
    
    rmse = sqrt(ssr / n_p);
    rse  = NaN; if n_p > k_val, rse = sqrt(ssr / (n_p - k_val)); end
    aicc = Inf; if n_p > k_val + 1, aicc = aic + (2*k_val*(k_val+1))/(n_p - k_val - 1); end
    dw   = durbinWatsonTest(res(:));
    
    %%% 4. CONSOLE REPORT %%%
    fprintf('\n--- Dataset %d/%d: %s ---\n', i, L, curve{i});
    fprintf('%-14s | %-6s | %-6s | %-6s | %-6s | %-5s | %-8s | %-8s | %-8s | %-6s | %-6s\n', ...
            'Model', 'AIC', 'AICc', 'RMSE', 'RSE', 'DW', 'N0', 'K', 'lambda', 'beta', 'sigma');
    fprintf('%s\n', repmat('-', 1, 100));
    
    b_str = '-'; if contains(lower(model_type), 'richards'), b_str = sprintf('%.4f', mle(5)); end
    sigma_val = mle(4); if contains(lower(model_type), 'logn'), sigma_val = exp(mle(4)); end
    
    fprintf('%-14s | %-6.1f | %-6.1f | %-6.2f | %-6.2f | %-5.2f | %-8.1f | %-8.1f | %-8.3f | %-6s | %-6.2f\n', ...
            model_type, aic, aicc, rmse, rse, dw, mle(3), mle(2), mle(1), b_str, sigma_val);
    
    %%% 5. PROFILE LIKELIHOOD ESTIMATION %%%
    fprintf('>> Profiling %d parameters...\n', num_active_params);
    
    prange_results = cell(1, num_active_params);
    norm_prof_results = cell(1, num_active_params);
    
    for p_idx_loop = 1:num_active_params
        param_index = active_idx(p_idx_loop);
        
        % Constraint Handling
        if param_index == 6 % Composite parameter: lambda * beta
            p_mle = mle(1) * mle(5);
            p_lb = p_mle * 0.1; 
            p_ub = p_mle * 5.0; 
        else
            p_lb = lb(param_index);
            p_ub = ub(param_index);
            p_mle = mle(param_index);
        end
        
        % Grid generation (dense around MLE)
        p_base = linspace(p_lb, p_ub, profile_npts);
        width = (p_ub - p_lb) * 0.05; 
        p_dense = linspace(p_mle - width, p_mle + width, 20);
        
        prange = sort([p_base, p_dense, p_mle]);
        prange = prange(prange >= p_lb & prange <= p_ub);
        prange = unique(prange); 
        n_pts_actual = length(prange);
        
        nll_profile = zeros(1, n_pts_actual);
        [~, mle_idx] = min(abs(prange - p_mle));
        
        % Leftward Optimization
        n0_nuisance = mle; 
        for k = mle_idx:-1:1
            p_val = prange(k);
            curr_lb = lb; curr_ub = ub;
            
            if param_index == 6
                curr_nonlcon = @(p) profile_product_nonlcon(p, nonlcon, p_val);
            else
                curr_lb(param_index) = p_val; curr_ub(param_index) = p_val; 
                n0_nuisance(param_index) = p_val;
                curr_nonlcon = nonlcon;
            end
            
            fun_nuisance = @(p_nuisance) likelihood_wrapper(model_type, p_nuisance, t, Ndata);
            [n_opt, nll_val] = fmincon(fun_nuisance, n0_nuisance, [],[],[],[], curr_lb, curr_ub, curr_nonlcon, options);
            
            nll_profile(k) = nll_val;
            n0_nuisance = n_opt; 
        end
        
        % Rightward Optimization
        n0_nuisance = mle; 
        for k = (mle_idx + 1):n_pts_actual
            p_val = prange(k);
            curr_lb = lb; curr_ub = ub;
            
            if param_index == 6
                curr_nonlcon = @(p) profile_product_nonlcon(p, nonlcon, p_val);
            else
                curr_lb(param_index) = p_val; curr_ub(param_index) = p_val;
                n0_nuisance(param_index) = p_val;
                curr_nonlcon = nonlcon;
            end
            
            fun_nuisance = @(p_nuisance) likelihood_wrapper(model_type, p_nuisance, t, Ndata);
            [n_opt, nll_val] = fmincon(fun_nuisance, n0_nuisance, [],[],[],[], curr_lb, curr_ub, curr_nonlcon, options);
            
            nll_profile(k) = nll_val;
            n0_nuisance = n_opt; 
        end
        
        norm_profile = nLL - nll_profile; 
        prange_results{p_idx_loop} = prange;
        norm_prof_results{p_idx_loop} = norm_profile;
    end
    
    %%% 6. PLOTTING %%%
    fig_name = sprintf('Profile_%s_%s_%s', folder_name_base, curve{i}, model_type);
    hFig = figure('Name', fig_name, 'NumberTitle', 'off', 'Position', fig_size);
    clf(hFig);
    
    for p_idx_loop = 1:num_active_params
        prange = prange_results{p_idx_loop};
        norm_profile = norm_prof_results{p_idx_loop};
        param_index = active_idx(p_idx_loop);
        plot_name = plot_param_names{p_idx_loop};
        
        if param_index == 6
            p_mle = mle(1) * mle(5);
            p_lb = min(prange); p_ub = max(prange);
        else
            p_mle = mle(param_index);
            p_lb = lb(param_index); p_ub = ub(param_index);
        end
        
        subplot(n_rows, n_cols, p_idx_loop); hold on;
        plot(prange, norm_profile, 'b-', 'LineWidth', 2.5); 
        yline(profile_TH, 'k--', '95% CI', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'FontSize', plot_fs-2);
        xline(p_mle, 'r-', 'MLE', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'FontSize', plot_fs-2);
        
        try
            [ci_low, ci_high] = interpolate_ci(prange, norm_profile, profile_TH);
            if ~isnan(ci_low), xline(ci_low, 'g:', 'LineWidth', 1.5); end
            if ~isnan(ci_high), xline(ci_high, 'g:', 'LineWidth', 1.5); end
        catch
            % Ignore if interpolate_ci is missing
        end
        
        title(sprintf('Profile: %s', plot_name), 'FontSize', plot_fs+2);
        xlabel(plot_name, 'FontSize', plot_fs, 'Interpreter', 'tex'); 
        
        % Add Y-label only to the first plot in the row to avoid clutter
        if p_idx_loop == 1
            ylabel('$\hat{\ell}$', 'FontSize', plot_fs, 'Interpreter', 'latex');
        end
        
        ylim([profile_TH * 2.5, 0.5]); 
        
        % Dynamic X-axis zoom
        zoom_th = profile_TH * 3; 
        idx_visible = find(norm_profile >= zoom_th);
        
        if length(idx_visible) > 1
            x_min = prange(idx_visible(1));
            x_max = prange(idx_visible(end));
            padding = (x_max - x_min) * 0.15; 
            if padding == 0, padding = abs(p_mle)*0.1; end
            xlim([max(p_lb, x_min - padding), min(p_ub, x_max + padding)]);
        else
            xlim([p_lb, p_ub]); 
        end
        
        set(gca, 'FontSize', plot_fs);
        grid on; box on;
    end
    
    sgtitle(sprintf('Profile Likelihoods - %s (%s)', curve{i}, model_type), 'FontWeight', 'bold', 'FontSize', plot_fs+4);
    
    % Save
    saveas(hFig, fullfile(save_dir, sprintf('%s.png', fig_name)));
    saveas(hFig, fullfile(save_dir, sprintf('%s.fig', fig_name)));
end

diary off;
fprintf('Profiling complete. Results and Log saved in "%s".\n', save_dir);

%%% HELPER FUNCTIONS %%%
function [c, ceq] = profile_product_nonlcon(p, orig_nonlcon, target_val)
    % Constrains lambda * beta = target_val
    if isempty(orig_nonlcon)
        c = [];
        ceq = p(1)*p(5) - target_val;
    else
        [c, ceq_orig] = orig_nonlcon(p);
        ceq = [ceq_orig(:); p(1)*p(5) - target_val];
    end
end