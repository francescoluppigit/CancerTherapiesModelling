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

% Active settings
data_folder  = data_folder10;
t_start      = 0; 
train_start  = 1; 
train_end    = 26; 
extra_days   = 10;
useGS        = false; 

% Directories and logging
[folder_path, folder_name_base, ~] = fileparts(data_folder);
if isempty(folder_name_base), [~, folder_name_base, ~] = fileparts(folder_path); end

save_dir = fullfile('img', folder_name_base);
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

log_filename = fullfile(save_dir, sprintf('Log_Forecast_%s_Tr%d-%d_%s.txt',...
    folder_name_base, train_start, train_end, datestr(now, 'yyyymmdd_HHMM')));
diary(log_filename);

fprintf('MLE Forecasting & Confidence Intervals\nDataset: %s\nTraining Window: [%d, %d]\nDate: %s\n\n', ...
    data_folder, train_start, train_end, datetime("now"));

%% 2. DATA IMPORT & MODEL PREP %%
[data_paths, curve, L] = load_data_set(data_folder); 

model_list0 = {'Logistic', 'Gompertz', 'Richards', ...
    'LogisticLogN', 'GompertzLogN', 'RichardsLogN'};
model_list1 = {'Logistic', 'Gompertz', 'Richards'};
model_list2 = {'LogisticLogN', 'GompertzLogN', 'RichardsLogN'};
model_list3 = {'Logistic', 'LogisticLogN'};
model_list4 = {'Gompertz', 'GompertzLogN'};
model_list5 = {'Richards', 'RichardsLogN'};

model_list = model_list0;

M = length(model_list); 

options = optimoptions('fmincon', 'Display', 'off', ...
    'MaxFunctionEvaluations', 5000, 'OptimalityTolerance', 1e-10);

colors = lines(M);
plot_fs = 18; 

%% 3. MAIN FORECAST LOOP %%
for i = 1:L
    [t_raw, Ndata_raw] = import_data(data_paths{i});
    
    idx_valid = (t_raw >= t_start);
    t_all = t_raw(idx_valid);
    Ndata_all = Ndata_raw(idx_valid);
    
    if length(t_all) < 4
        fprintf('Dataset %s skipped (insufficient points).\n', curve{i});
        continue;
    end
    
    % Data split
    idx_pre   = (t_all < train_start);
    idx_train = (t_all >= train_start) & (t_all <= train_end);
    idx_test  = (t_all > train_end);
    
    t_pre   = t_all(idx_pre);   Ndata_pre   = Ndata_all(idx_pre);
    t_train = t_all(idx_train); Ndata_train = Ndata_all(idx_train);
    t_test  = t_all(idx_test);  Ndata_test  = Ndata_all(idx_test);
    
    n_tot = length(t_all); n_tr = length(t_train); n_te = length(t_test);
    
    if n_tr < 5
        fprintf('Dataset %s skipped (insufficient training data).\n', curve{i});
        continue;
    end
    
    max_t = max(t_all);
    t_sim = linspace(t_start, max_t + extra_days, 200)'; 
    
    metrics_eval = nan(M, 4); % [RMSE_Tr, RMSE_Te, MAPE_Tr, MAPE_Te]
    dw_vec       = nan(M, 1);
    params_mat   = nan(M, 5);
    ip_vec       = nan(M, 1);
    sim_curves   = cell(M, 1);
    all_res_cell = cell(M, 1);
    
    for j = 1:M
        md = lower(model_list{j});
        try
            % Usa Ndata_all per fornire stime di K e N0 coerenti con i limiti globali
            [l0, K0, N00, s0, b0, ~, lb, ub] = initialise_param(md, Ndata_all);
            
            % Esegui l'ottimizzazione ESCLUSIVAMENTE sui dati di training, passando []
            if useGS
                [mle, nLL] = fit_growth_model_global(md, t_train, Ndata_train, [l0,K0,N00,s0,b0], lb, ub, [], options);
            else
                [mle, nLL] = fit_growth_model_local(md, t_train, Ndata_train, [l0,K0,N00,s0,b0], lb, ub, [], options);
            end
            
            params_mat(j, 1:length(mle)) = mle;
            ip_vec(j) = ip_fun(mle, md);
            
            % Training Eval
            [~, ssr_tr, ~, ~, ~, mape_tr] = evaluate_fit(md, mle, t_train, Ndata_train, nLL);
            metrics_eval(j, 1) = sqrt(ssr_tr / n_tr);
            metrics_eval(j, 3) = mape_tr;
            
            % Test Eval
            if n_te > 0
                [~, ssr_te, ~, ~, ~, mape_te] = evaluate_fit(md, mle, t_test, Ndata_test, 0);
                metrics_eval(j, 2) = sqrt(ssr_te / n_te);
                metrics_eval(j, 4) = mape_te;
            end
            
            % Global Residuals & Curve
            [~, ~, ~, ~, res_all, ~] = evaluate_fit(md, mle, t_all, Ndata_all, 0);
            all_res_cell{j} = res_all;
            dw_vec(j) = durbinWatsonTest(res_all(:));
            
            [~, ~, ~, N_sim_plot, ~, ~] = evaluate_fit(md, mle, t_sim, zeros(size(t_sim)), 0);
            sim_curves{j} = N_sim_plot;
        catch
            % Fit failed
        end
    end
    
    %% 4. REPORT %%
    [best_val, win_idx] = min(metrics_eval(:, 2)); 
    if isnan(best_val), [best_val, win_idx] = min(metrics_eval(:, 1)); end
    
    if isinf(best_val) || isnan(best_val)
        fprintf('Dataset %s: Fit failed for all models.\n', curve{i});
        continue;
    end
    
    fprintf('\n--- Dataset %d/%d: %s ---\n', i, L, curve{i});
    fprintf('>> BEST FORECAST: %s\n\n', upper(model_list{win_idx}));
    
    fprintf('%-14s | %-8s | %-7s | %-7s | %-7s | %-7s | %-5s | %-8s | %-8s | %-8s | %-6s | %-6s | %-6s\n', ...
        'Model', 'Δ_RMSE_Te', 'RMSE_Tr', 'RMSE_Te', 'MAPE_Tr', 'MAPE_Te', 'DW', 'N0', 'K', 'lambda', 'beta', 'sigma', 'IP');
    fprintf('%s\n', repmat('-', 1, 130));
    
    for j = 1:M
        prm = params_mat(j, :);
        if isnan(prm(1))
            fprintf('%-14s | FIT FAILED\n', model_list{j});
            continue;
        end
        
        val = metrics_eval(j, 2); 
        if isnan(val), val = metrics_eval(j, 1); end 
        delta = val - best_val;
        perc = 0; if abs(best_val) > 1e-6, perc = (delta / abs(best_val)) * 100; end
        
        r_te_str = '-'; if ~isnan(metrics_eval(j,2)), r_te_str = sprintf('%.2f', metrics_eval(j,2)); end
        m_te_str = '-'; if ~isnan(metrics_eval(j,4)), m_te_str = sprintf('%.1f%%', metrics_eval(j,4)); end
        ip_str   = '-'; if ~isnan(ip_vec(j)), ip_str = sprintf('%.1f', ip_vec(j)); end
        
        b_str = '-'; if contains(lower(model_list{j}), 'richards'), b_str = sprintf('%.3f', prm(5)); end
        sigma_val = prm(4); if contains(lower(model_list{j}), 'logn'), sigma_val = exp(prm(4)); end
        
        fprintf('%-14s | +%-6.1f%% | %-7.2f | %-7s | %-6.1f%% | %-7s | %-5.2f | %-8.1f | %-8.1f | %-8.3f | %-6s | %-6.2f | %-6s\n', ...
            model_list{j}, perc, metrics_eval(j,1), r_te_str, metrics_eval(j,3), m_te_str, dw_vec(j), ...
            prm(3), prm(2), prm(1), b_str, sigma_val, ip_str);
    end
    fprintf('\n');

    %% 5. PLOT %%
    fig_name = sprintf('Forecast_%s_%s', folder_name_base, curve{i});
    hFig = figure('Name', fig_name, 'NumberTitle', 'off', 'Position', [100, 100, 2844, 800]); 
    
    % Subplot grid dimensions (max 3 columns)
    num_cols = min(M, 3);
    num_rows = ceil(M / 3);
    
    for j = 1:M
        subplot(num_rows, num_cols, j); hold on; 
        
        if isnan(metrics_eval(j, 1)), continue; end
        
        sigma_est = params_mat(j, 4);
        N_sim = sim_curves{j}(:);
        
        % 1-sigma and 2-sigma bands calculation
        if contains(lower(model_list{j}), 'logn')
            CI1_up = N_sim .* exp(1 * sigma_est);
            CI1_lo = N_sim ./ exp(1 * sigma_est);
            CI2_up = N_sim .* exp(2 * sigma_est);
            CI2_lo = N_sim ./ exp(2 * sigma_est);
        else
            CI1_up = N_sim + 1 * sigma_est;
            CI1_lo = max(0, N_sim - 1 * sigma_est); 
            CI2_up = N_sim + 2 * sigma_est;
            CI2_lo = max(0, N_sim - 2 * sigma_est); 
        end
        
        % Plot Data
        %if ~isempty(t_pre), plot(t_pre, Ndata_pre, 'o', 'Color', [0.6 0.6 0.6], 'MarkerFaceColor', [0.6 0.6 0.6], 'MarkerSize', 5, 'DisplayName', 'Pre-Train'); end
        plot(t_train, Ndata_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, 'DisplayName', 'Train Data');
        if ~isempty(t_test), plot(t_test, Ndata_test, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6, 'DisplayName', 'Test Data'); end

        t_sim_row = t_sim(:)';
        
        % Logic to show bands only from train_end onwards
        idx_fore  = t_sim >= train_end;
        t_fore    = t_sim(idx_fore)';
        CI1_up_f  = CI1_up(idx_fore)';
        CI1_lo_f  = CI1_lo(idx_fore)';
        CI2_up_f  = CI2_up(idx_fore)';
        CI2_lo_f  = CI2_lo(idx_fore)';
        
        % Plot 2-Sigma Confidence Band (Outer)
        fill([t_fore, fliplr(t_fore)], [CI2_up_f, fliplr(CI2_lo_f)], ...
             colors(j,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
         
        % Plot 1-Sigma Confidence Band (Inner)
        fill([t_fore, fliplr(t_fore)], [CI1_up_f, fliplr(CI1_lo_f)], ...
             colors(j,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');

        % Plot Inflection Point (IP)
        ip_val = ip_vec(j);
        if ~isnan(ip_val) && ip_val >= min(t_sim) && ip_val <= max(t_sim)
            xline(ip_val, ':', 'IP', 'Color', colors(j,:), 'LineWidth', 1.5, ...
                  'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'left', ...
                  'FontSize', 10, 'HandleVisibility', 'off');
        end

        % Plot Curve
        plot(t_sim, N_sim, '-', 'LineWidth', 2, 'Color', colors(j,:), 'DisplayName', 'Forecast');

        title(['\bf ' model_list{j} ' - ' curve{i}], 'FontSize', plot_fs);
        
        xline(train_start, 'k--','LineWidth', 1.5, 'HandleVisibility', 'off', 'FontSize', 10);
        xline(train_end, 'k--','LineWidth', 1.5, 'HandleVisibility', 'off', 'FontSize', 10);
        
        xlabel('Time (days)', 'FontSize', plot_fs); 
        
        % Only show Y-label on the first column to avoid clutter
        if mod(j-1, num_cols) == 0
            ylabel('Tumour Volume (mm^3)', 'FontSize', plot_fs); 
        end
        
        xlim([t_start - 1, max_t + 11]);
        max_y_data = max(Ndata_all) * 1.2;
        max_y_plot = min(max(CI2_up(t_sim <= max_t+extra_days)), max_y_data * 1.2);
        ylim([0, max(max_y_data, max_y_plot)]);
        
        set(gca, 'FontSize', plot_fs);
        grid on; box on;
        
        % Only show Legend on the first subplot
        if j == 1, legend('Location', 'SouthEast', 'FontSize', plot_fs-2); end
    end
    
    % Save
    saveas(hFig, fullfile(save_dir, sprintf('%s.png', fig_name)));
    saveas(hFig, fullfile(save_dir, sprintf('%s.fig', fig_name)));
end

diary off;
fprintf('Analysis complete. Results and Log saved in "%s".\n', save_dir);