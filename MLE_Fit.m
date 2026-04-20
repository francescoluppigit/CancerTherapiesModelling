%% 1. CONFIGURATION & SETUP %%
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


data_folder = data_folder15;
t_start = 0;

qq_flag = false;
metric_selection = 'AICc'; % Options: 'AIC', 'AICc', 'RMSE', 'RSE', 'MAPE'
useGS = false; 

% Directories and logging
[folder_path, folder_name_base, ~] = fileparts(data_folder);
if isempty(folder_name_base), [~, folder_name_base, ~] = fileparts(folder_path); end
save_dir = fullfile('img', folder_name_base);
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

% Route diary output to the 'img' directory
log_filename = fullfile(save_dir, sprintf('Log_ModelFit_%s_%s.txt', folder_name_base, datestr(now, 'yyyymmdd_HHMM')));
diary(log_filename);

fprintf('MLE Model Comparison & Diagnostics\nDataset: %s\nCriterion: %s\nDate: %s\n\n', ...
    data_folder, metric_selection, datetime("now"));

%% 2. DATA IMPORT %%
[data_paths, curve, L] = load_data_set(data_folder); 

time_intervals = cell(L, 1);
for i = 1:L, time_intervals{i} = [t_start]; end

model_list0 = {'Logistic', 'LogisticLogN',...
    'Gompertz', 'GompertzLogN',...
    'Richards', 'RichardsLogN'};
model_list1 = {'Logistic', 'Gompertz', 'Richards'};
model_list2 = {'LogisticLogN', 'GompertzLogN', 'RichardsLogN'};
model_list3 = {'Logistic', 'LogisticLogN'};
model_list4 = {'Gompertz', 'GompertzLogN'};
model_list5 = {'Richards', 'RichardsLogN'};
model_list6 = {'Logistic', 'Gompertz', 'Richards',...
    'LogisticLogN', 'GompertzLogN', 'RichardsLogN'};

model_list = model_list1;

M = length(model_list); 

% Optimiser setup
options = optimoptions('fmincon', 'Display', 'off', ...
    'MaxFunctionEvaluations', 5000, 'OptimalityTolerance', 1e-10);

% Plotting parameters
colors = lines(M);
markers = {'o', 's', '^', 'd'}; 
plot_fs = 18; 

%% 3. MAIN LOOP %%
metric_idx_map = struct('AIC', 1, 'AICc', 2, 'RMSE', 3, 'RSE', 4, 'MAPE', 5);
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
    if n_p < 6
        fprintf('Dataset %s skipped (insufficient data).\n', curve{i});
        continue;
    end
    
    metrics_mat = nan(M, 5); 
    dw_vec      = nan(M, 1);
    params_mat  = nan(M, 5);
    ip_vec      = nan(M, 1); 
    
    nmle_cell   = cell(M, 1);
    res_cell    = cell(M, 1);
    
    for j = 1:M
        md = lower(model_list{j});
        try
            [l0, K0, N00, s0, b0, ~, lb, ub] = initialise_param(md, Ndata);
            
            if useGS
                [mle, nLL] = fit_growth_model_global(md, t, Ndata, [l0,K0,N00,s0,b0], lb, ub, [], options);
            else
                [mle, nLL] = fit_growth_model_local(md, t, Ndata, [l0,K0,N00,s0,b0], lb, ub, [], options);
            end
            
            [k_val, ssr, aic, Nmle, res, mape] = evaluate_fit(md, mle, t, Ndata, nLL);
            
            rmse_val = sqrt(ssr / n_p);
            rse_val  = NaN; if n_p > k_val, rse_val = sqrt(ssr / (n_p - k_val)); end
            aicc_val = Inf; if n_p > k_val + 1, aicc_val = aic + (2*k_val*(k_val+1))/(n_p - k_val - 1); end
            
            metrics_mat(j, :) = [aic, aicc_val, rmse_val, rse_val, mape];
            dw_vec(j)         = durbinWatsonTest(res(:));
            params_mat(j, 1:length(mle)) = mle;
            nmle_cell{j}      = Nmle;
            res_cell{j}       = res;
            ip_vec(j)         = ip_fun(mle, md);
        catch
            % Fit failed - NaNs handled in the report
        end
    end
    
    %% 4. REPORT & PLOT %%
    [best_val, win_idx] = min(metrics_mat(:, sel_idx));
    if isinf(best_val) || isnan(best_val)
        fprintf('Dataset %s: Fit failed for all models.\n', curve{i});
        continue;
    end
    
    fprintf('\n--- Dataset %d/%d: %s ---\n', i, L, curve{i});
    fprintf('>> BEST: %s (Metric: %s)\n\n', model_list{win_idx}, metric_selection);
    
    fprintf('%-14s | %-6s | %-6s | %-6s | %-6s | %-6s | %-7s | %-5s | %-8s | %-8s | %-8s | %-6s | %-6s | %-6s | %-6s\n', ...
            'Model', ['Δ_', metric_selection], 'AIC', 'AICc', 'RMSE', 'RSE', 'MAPE', 'DW', 'N0', 'K', 'lambda', 'beta', 'sigma', 'CV', 'IP');
    fprintf('%s\n', repmat('-', 1, 140));
    
    max_res_norm = 0; 

    for j = 1:M
        prm = params_mat(j, :);
        if isnan(prm(1))
            fprintf('%-14s | FIT FAILED\n', model_list{j});
            continue;
        end

        res_norm = res_cell{j} / prm(4);
        max_res_norm = max(max_res_norm, max(abs(res_norm)));

        val = metrics_mat(j, sel_idx);
        delta = val - best_val;
        perc = 0; if abs(best_val) > 1e-6, perc = (delta / abs(best_val)) * 100; end
        
        cv_str = '-';
        if contains(lower(model_list{j}), 'logn')
            cv_val = sqrt(exp(prm(4)^2)-1)*100; 
            cv_str = sprintf('%.1f%%', cv_val); 
            sigma_val = exp(prm(4));
        else
            sigma_val = prm(4); 
        end
        
        b_str = '-'; 
        if contains(lower(model_list{j}), 'richards'), b_str = sprintf('%.3f', prm(5)); end
        ip_str = '-'; 
        if ~isnan(ip_vec(j)), ip_str = sprintf('%.1f', ip_vec(j)); end
        
        fprintf('%-14s | %-4.1f%%  | %-6.1f | %-6.1f | %-6.2f | %-6.2f | %-6.1f%% | %-5.2f | %-8.1f | %-8.1f | %-8.3f | %-6s | %-6.2f | %-6s | %-6s\n', ...
                model_list{j}, perc, metrics_mat(j, 1), metrics_mat(j, 2), metrics_mat(j, 3), metrics_mat(j, 4), ...
                metrics_mat(j, 5), dw_vec(j), prm(3), prm(2), prm(1), b_str, sigma_val, cv_str, ip_str);
    end
    fprintf('\n');

    % Setup main figure
    fig_name = sprintf('Fit_%s_%s', folder_name_base, curve{i});
    hFig = figure('Name', fig_name, 'NumberTitle', 'off', 'Position', [100, 100, 1422, 800]); 
    
    res_lim = max(3, ceil(max_res_norm * 1.2)); 
    
    subplot(2,1,1); hold on;
    plot(t, Ndata, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, 'DisplayName', 'Data');

    for j = 1:M
        if isnan(metrics_mat(j, 1)), continue; end 
        
        c = colors(j, :);
        mkr = markers{mod(j-1, length(markers))+1};
        
        % Dynamic legends
        % fit_leg_str = sprintf('%s (%s=%.1f)', model_list{j}, metric_selection, metrics_mat(j, sel_idx));
        fit_leg_str = sprintf('%s', model_list{j});
        res_leg_str = sprintf('%s', model_list{j});
        % res_leg_str = sprintf('%s (DW=%.2f)', model_list{j}, dw_vec(j));
        
        % Subplot 1: Fit
        subplot(2,1,1); hold on;
        plot(t, nmle_cell{j}, '-', 'LineWidth', 2, 'Color', c, 'DisplayName', fit_leg_str);
        
        if ~isnan(ip_vec(j)) && ip_vec(j) >= min(t) && ip_vec(j) <= max(t)
            xline(ip_vec(j), '--', 'IP', 'Color', c, 'LineWidth', 1.5, ...
                  'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'center', ...
                  'FontSize', 10, 'HandleVisibility', 'off');
        end
        
        % Subplot 2: Normalised Residuals
        subplot(2,1,2); hold on;
        res_norm = res_cell{j} / params_mat(j, 4); 
        plot(t, res_norm, ['-', mkr], 'LineWidth', 1.5, 'Color', c, ...
             'MarkerFaceColor', c, 'MarkerSize', 6, 'DisplayName', res_leg_str);
         
        if ~isnan(ip_vec(j)) && ip_vec(j) >= min(t) && ip_vec(j) <= max(t)
            xline(ip_vec(j), '--', 'IP', 'Color', c, 'LineWidth', 1.5, ...
                  'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'center', ...
                  'FontSize', 10, 'HandleVisibility', 'off');
        end
    end
    
    % Subplot 1 format
    subplot(2,1,1);
    title(sprintf('Fit - %s', curve{i}), 'FontSize', plot_fs + 2);
    ylabel('Tumour Volume (mm^3)', 'FontSize', plot_fs+4); 
    xlabel('Time (days)', 'FontSize', plot_fs+4);
    xlim([max(0, t_start-1), max(t)+1]); ylim([0, max(Ndata)*1.2]);
    set(gca, 'FontSize', plot_fs);
    legend('Location', 'bestoutside', 'FontSize', plot_fs); grid on; box on;
    
    % Subplot 2 format
    subplot(2,1,2);
    yline(0, 'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    title(sprintf('Normalised Residuals - %s', curve{i}), 'FontSize', plot_fs + 2);
    ylabel('Residuals', 'FontSize', plot_fs+4); 
    xlabel('Time (days)', 'FontSize', plot_fs+4);
    xlim([max(0, t_start-1), max(t)+1]); ylim([-res_lim, res_lim]);
    set(gca, 'FontSize', plot_fs);
    legend('Location', 'bestoutside', 'FontSize', plot_fs); grid on; box on;
    
    saveas(hFig, fullfile(save_dir, sprintf('%s.png', fig_name)));
    saveas(hFig, fullfile(save_dir, sprintf('%s.fig', fig_name)));
    
    % QQ-Plot setup
    if qq_flag
        fig_name_qq = sprintf('QQ_Plot_%s_%s', folder_name_base, curve{i});
        hFigQQ = figure('Name', fig_name_qq, 'NumberTitle', 'off', 'Position', [150, 150, 500*M, 500]); 
    
        for j = 1:M
            if isnan(metrics_mat(j, 1)), continue; end 
            subplot(1, M, j); hold on;
    
            c = colors(j, :);
            res_norm = res_cell{j} / params_mat(j, 4);
            n_res = length(res_norm);
            sorted_res = sort(res_norm); 
            Pi = ((1:n_res) - 0.5) / n_res;
            theo_quantiles = norminv(Pi, 0, 1); 
    
            min_val = min([theo_quantiles, sorted_res']);
            max_val = max([theo_quantiles, sorted_res']);
            ax_limits = [floor(min_val)-0.5, ceil(max_val)+0.5];
    
            plot(ax_limits, ax_limits, 'k--', 'LineWidth', 1.5, 'DisplayName', 'N(0,1)');
            plot(theo_quantiles, sorted_res, 'o', 'MarkerEdgeColor', c, ...
                 'MarkerFaceColor', c, 'MarkerSize', 6, 'DisplayName', 'Residuals');
    
            title(sprintf('Q-Q Plot: %s', model_list{j}), 'FontSize', plot_fs);
            xlabel('Theoretical Quantiles', 'FontSize', plot_fs+4);
            ylabel('Sample Quantiles', 'FontSize', plot_fs+4);
            xlim(ax_limits); ylim(ax_limits);
            axis square; grid on; box on;
            set(gca, 'FontSize', plot_fs);
            legend('Location', 'northwest', 'FontSize', plot_fs);
        end
        saveas(hFigQQ, fullfile(save_dir, sprintf('%s.png', fig_name_qq)));
    end

end

diary off;
fprintf('Analysis complete. Results and Log saved in "%s" directory.\n', save_dir);