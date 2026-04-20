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

data_folder22 = 'Dati3\cyclic'; data_folder23 = 'Dati3\neutropenia';

% Active settings
data_folder = data_folder1;
t_start = 11;

metric_selection = 'AICc'; % Options: 'AIC', 'AICc', 'RMSE', 'RSE'
useGS = false; 

% Directories and logging
[folder_path, folder_name_base, ~] = fileparts(data_folder);
if isempty(folder_name_base), [~, folder_name_base, ~] = fileparts(folder_path); end

save_dir = fullfile('img', folder_name_base);
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

log_filename = fullfile(save_dir, sprintf('Log_Comparison_%s_%s.txt', metric_selection, datestr(now, 'yyyymmdd_HHMM')));
diary(log_filename);

fprintf('MLE Models Comparison & Selection\nDataset: %s\nCriterion: %s\nDate: %s\n\n', ...
    data_folder, metric_selection, datetime("now"));

%%% 2. DATA IMPORT %%%
[data_paths, curve, L] = load_data_set(data_folder); 

time_intervals = cell(L, 1);
for i = 1:L, time_intervals{i} = [t_start]; end

model_list0 = {'Logistic', 'Gompertz', 'Richards',...
    'LogisticLogN', 'GompertzLogN', 'RichardsLogN'};
model_list1 = {'Logistic', 'Gompertz', 'Richards'};

model_list = model_list1;

M = length(model_list); 

% Optimiser setup
options = optimoptions('fmincon', 'Display', 'off', ...
    'MaxFunctionEvaluations', 5000, 'OptimalityTolerance', 1e-10);

% Plotting parameters
colors_ds = lines(L); % Unique colors per dataset
plot_fs = 20; 

% Global figure initialization
hFigFit = figure('Name', 'Best Fits Comparison', 'NumberTitle', 'off', 'Position', [100, 100, 900, 700]); 
hold on;
hFigRes = figure('Name', 'Residuals Comparison', 'NumberTitle', 'off', 'Position', [1050, 100, 600, max(600, L*150)]); 

%%% 3. MAIN ESTIMATION LOOP %%%
metric_idx_map = struct('AIC', 1, 'AICc', 2, 'RMSE', 3, 'RSE', 4);
sel_idx = metric_idx_map.(metric_selection);

for i = 1:L
    [t, Ndata] = import_data(data_paths{i});
    
    if ~isempty(time_intervals{i})
        t_int = time_intervals{i};
        te = max(t); if length(t_int) > 1, te = t_int(2); end
        idx = (t >= t_int(1)) & (t <= te);
        t = t(idx); Ndata = Ndata(idx);
    end
    
    n_p = length(t);
    if n_p < 5
        fprintf('Dataset %s skipped (insufficient data).\n', curve{i});
        continue;
    end
    
    metrics_matrix = nan(M, 4); 
    dw_vector      = nan(M, 1);
    params_cell    = cell(M, 1);
    res_cell       = cell(M, 1);
    nmle_cell      = cell(M, 1);
    
    for j = 1:M
        md = lower(model_list{j});
        try
            [l0, K0, N00, s0, b0, ~, lb, ub] = initialise_param(md, Ndata);
            
            if useGS
                [mle, nLL] = fit_growth_model_global(md, t, Ndata, [l0,K0,N00,s0,b0], lb, ub, [], options);
            else
                [mle, nLL] = fit_growth_model_local(md, t, Ndata, [l0,K0,N00,s0,b0], lb, ub, [], options);
            end
            
            [k_val, ssr, aic, Nmle, res, ~] = evaluate_fit(md, mle, t, Ndata, nLL);
            
            rmse_val = sqrt(ssr / n_p);
            rse_val  = NaN; if n_p > k_val, rse_val = sqrt(ssr / (n_p - k_val)); end
            aicc_val = Inf; if n_p > k_val + 1, aicc_val = aic + (2*k_val*(k_val+1))/(n_p - k_val - 1); end
            
            metrics_matrix(j, :) = [aic, aicc_val, rmse_val, rse_val];
            dw_vector(j) = durbinWatsonTest(res(:));
            params_cell{j} = mle;
            res_cell{j} = res;
            nmle_cell{j} = Nmle;
        catch
            % Fit failed
        end
    end
    
    %%% 4. SELECTION & REPORTING %%%
    [best_val, win_idx] = min(metrics_matrix(:, sel_idx));
    
    if isinf(best_val) || isnan(best_val)
        fprintf('Dataset %s: Fit failed for all models.\n', curve{i});
        continue;
    end
    
    win_model = model_list{win_idx};
    win_prm = params_cell{win_idx};
    win_res = res_cell{win_idx};
    win_nmle = nmle_cell{win_idx};
    
    fprintf('\n--- Dataset %d/%d: %s ---\n', i, L, curve{i});
    fprintf('>> WINNER: %s (Criterion: %s)\n\n', upper(win_model), metric_selection);
    
    fprintf('%-12s | %-8s | %-6s | %-6s | %-6s | %-6s | %-5s | %-8s | %-8s | %-8s | %-6s | %-6s\n', ...
            'Model', ['Δ_', metric_selection], 'AIC', 'AICc', 'RMSE', 'RSE', 'DW', 'N0', 'K', 'lambda', 'beta', 'sigma');
    fprintf('%s\n', repmat('-', 1, 110));

    for j = 1:M
        prm = params_cell{j};
        if isempty(prm) || isnan(prm(1))
            fprintf('%-12s | FIT FAILED\n', model_list{j});
            continue;
        end

        if contains(lower(model_list{j}), 'logn')
            sigma_val = exp(prm(4));
        else
            sigma_val = prm(4); 
        end
        
        val = metrics_matrix(j, sel_idx);
        delta = val - best_val;
        perc = 0; if abs(best_val) > 1e-6, perc = (delta / abs(best_val)) * 100; end
        
        b_str = '-';
        if length(prm) >= 5, b_str = sprintf('%.3f', prm(5)); end
        
        fprintf('%-12s | %-4.1f%%   | %-6.1f | %-6.1f | %-6.2f | %-6.2f | %-5.2f | %-8.1f | %-8.1f | %-8.3f | %-6s | %-6.2f\n', ...
                model_list{j}, perc, metrics_matrix(j, 1), metrics_matrix(j, 2), metrics_matrix(j, 3), metrics_matrix(j, 4), ...
                dw_vector(j), prm(3), prm(2), prm(1), b_str, sigma_val);
    end
    fprintf('\n');

    %%% 5. PLOTTING %%%
    c = colors_ds(i, :);
    
    % Figure 1: Best Fit Overlay
    figure(hFigFit);
    plot_label = sprintf('%s (%s)', curve{i}, win_model);
    plot(t, Ndata, 'o', 'MarkerFaceColor', c, 'MarkerEdgeColor', 'k', 'MarkerSize', 7, 'HandleVisibility', 'off');
    plot(t, win_nmle, '-', 'LineWidth', 2.5, 'Color', c, 'DisplayName', plot_label);
    
    % Figure 2: Residuals Subplot
    figure(hFigRes);
    subplot(L, 1, i); hold on;
    yline(0, 'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    
    res_norm = win_res / win_prm(4);
    res_lim = max(3, ceil(max(abs(res_norm)) * 1.2)); % Dynamic limits per dataset
    
    plot(t, res_norm, '-o', 'LineWidth', 1.5, 'Color', c, 'MarkerFaceColor', c, 'MarkerSize', 6);
    title(sprintf('%s', plot_label), 'FontSize', plot_fs);
    ylabel('Residuals', 'FontSize', plot_fs - 2); 
    xlim([max(0, t_start-1), max(t)+5]); ylim([-res_lim, res_lim]);
    set(gca, 'FontSize', plot_fs - 2);
    grid on; box on;
    
    if i == L, xlabel('Time (days)', 'FontSize', plot_fs); end
end

% Finalize Figure 1
figure(hFigFit);
title(sprintf('Best Fits Comparison - Metric: %s', metric_selection), 'FontSize', plot_fs + 2);
ylabel('Tumour Volume (mm^3)', 'FontSize', plot_fs); 
xlabel('Time (days)', 'FontSize', plot_fs);
xlim([max(0, t_start-1), Inf]); ylim([0, Inf]);
set(gca, 'FontSize', plot_fs);
legend('Location', 'best', 'FontSize', plot_fs - 2); grid on; box on;

% Save Outputs
saveas(hFigFit, fullfile(save_dir, sprintf('BestFits_%s.png', folder_name_base)));
saveas(hFigFit, fullfile(save_dir, sprintf('BestFits_%s.fig', folder_name_base)));
saveas(hFigRes, fullfile(save_dir, sprintf('Residuals_%s.png', folder_name_base)));
saveas(hFigRes, fullfile(save_dir, sprintf('Residuals_%s.fig', folder_name_base)));

diary off;
fprintf('Analysis complete. Results and Log saved in "%s" directory.\n', save_dir);