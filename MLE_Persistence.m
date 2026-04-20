%% ============================ CONFIGURATION =============================
clear; clc; close all;

%% ============================= DATA IMPORT ==============================
% --- 1. Data Folder & General Config ---
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

data_folder = data_folder1;
t_start = 11;
N_RUNS = 499;

[data_paths, curve, L] = load_data_set(data_folder); 

%% ============================= INTERVALS & MODELS =======================
time_intervals = cell(L, 1);
for i = 1:L, time_intervals{i} = [t_start]; end

% Model selection
model_list0 = {'Logistic', 'Gompertz', 'Richards',...
    'LogisticLogN', 'GompertzLogN', 'RichardsLogN'};
model_list1 = {'Logistic', 'Gompertz', 'Richards'};

model_list = model_list0;
M = length(model_list); 
useGS = false; 

%% ====================== MC ITERATION SETTINGS ===========================
do_subsampling = true;
useParallel = true; 
options = optimoptions('fmincon','Display','off',...
    'MaxFunctionEvaluations',5000,'OptimalityTolerance',1e-10);

%% --- Output Directory & Logging Setup ---
% Estrai il nome del dataset dalla cartella
[~, dataset_name, ~] = fileparts(data_folder);
if isempty(dataset_name)
    [~, dataset_name, ~] = fileparts(data_folder(1:end-1));
end

% Crea la cartella img/nome_folder se non esiste
save_dir = fullfile('img', dataset_name);
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

log_filename = fullfile(save_dir, sprintf('Persistence_%s_Log_%d.txt', dataset_name, N_RUNS));
diary(log_filename);
fprintf('--- Starting Persistence Analysis ---\n');
fprintf('Date/Time: %s\n\n', datestr(now));

% Inizializzazione dati per esportazione CSV
csv_header = {'Dataset', 'Pts_Removed', 'Pct_Removed', 'Model', ...
              'Win_AICc_pct', 'Win_AIC_pct', 'Win_RSE_pct', ...
              'Mean_AICc', 'Mean_AIC', 'Mean_RSE', 'Mean_DW', 'Mean_MAPE', ...
              'lambda', 'K', 'N0', 'sigma', 'beta', 'lambda_beta'};
csv_data = {};

%% ======================= DATA PRE-LOADING ===============================
fprintf('Pre-loading %d datasets...\n', L);
all_t_data = cell(L, 1);
all_Ndata_data = cell(L, 1);
N_original_per_dataset = zeros(L, 1);

for i = 1:L
    [all_t_data{i}, all_Ndata_data{i}] = import_data(data_paths{i});
    t_temp = all_t_data{i}; curr_int = time_intervals{i};
    if ~isempty(curr_int)
        if length(curr_int)==1, t_end=max(t_temp); else, t_end=curr_int(2); end
        idx = t_temp >= curr_int(1) & t_temp <= t_end;
        N_original_per_dataset(i) = sum(idx);
    else
        N_original_per_dataset(i) = length(t_temp);
    end
end

%% ======================= REMOVAL PLANNING ===============================
Removal_Plan = cell(L, 1);
MIN_PTS = 7; 
for i = 1:L
    N = N_original_per_dataset(i);
    max_removable = max(0, N - MIN_PTS); 
    if N < 20
        removals = 0 : 1 : max_removable;
    else
        target_pcts = 0 : 0.05 : 0.90; 
        removals = unique(round(target_pcts * N));
        removals = removals(removals <= max_removable);
    end
    Removal_Plan{i} = removals;
end
Max_Steps = max(cellfun(@length, Removal_Plan));

%% ========================= STORAGE INITIALIZATION =======================
Persist_AICc  = nan(L, M, Max_Steps); 
Persist_AIC   = nan(L, M, Max_Steps); 
Persist_RSE   = nan(L, M, Max_Steps); 
Mean_AICc     = nan(L, M, Max_Steps); 
Mean_AIC      = nan(L, M, Max_Steps); 
Mean_RSE      = nan(L, M, Max_Steps); 
Mean_DW       = nan(L, M, Max_Steps); 
Mean_MAPE     = nan(L, M, Max_Steps); 
Mean_Params   = nan(L, M, 5, Max_Steps); 
Actual_Pct    = nan(L, Max_Steps);    

if useParallel
    p = gcp('nocreate');
    if isempty(p), try, parpool('Processes'); catch, parpool; end; end
end

%% ======================= OUTER LOOP =====================================
fprintf('\nStarting Analysis over %d points ...\n', max(N_original_per_dataset));
fprintf('\n%d steps, %d MC runs/step...\n', Max_Steps, N_RUNS);
tic; 

for step_k = 1 : Max_Steps
    
    current_n_remove_vec = nan(L, 1);
    has_work = false;
    for i = 1:L
        if step_k <= length(Removal_Plan{i})
            n_rem = Removal_Plan{i}(step_k);
            current_n_remove_vec(i) = n_rem;
            Actual_Pct(i, step_k) = n_rem / N_original_per_dataset(i);
            has_work = true;
        end
    end
    if ~has_work, break; end
    
    fprintf('\n========================================================================\n');
    fprintf('--- PROCESSING STEP %d/%d ---\n', step_k, Max_Steps);
    
    Temp_AICc   = nan(L, M, N_RUNS); 
    Temp_AIC    = nan(L, M, N_RUNS); 
    Temp_RSE    = nan(L, M, N_RUNS); 
    Temp_DW     = nan(L, M, N_RUNS);
    Temp_MAPE   = nan(L, M, N_RUNS);
    Temp_Params = nan(L, M, 5, N_RUNS);
    
    is_baseline = all(current_n_remove_vec(~isnan(current_n_remove_vec)) == 0);
    
    if is_baseline
        [aicc, aic, rse, dw, mape, prm] = run_mc_step(L, all_t_data, all_Ndata_data, time_intervals, ...
                                       do_subsampling, current_n_remove_vec, model_list, M, useGS, options);
        for r = 1:N_RUNS
            Temp_AICc(:,:,r)=aicc; Temp_AIC(:,:,r)=aic; Temp_RSE(:,:,r)=rse; Temp_DW(:,:,r)=dw; Temp_MAPE(:,:,r)=mape; Temp_Params(:,:,:,r)=prm;
        end
    else 
        if useParallel
            parfor run_idx = 1:N_RUNS
                [aicc, aic, rse, dw, mape, prm] = run_mc_step(L, all_t_data, all_Ndata_data, time_intervals, ...
                                               do_subsampling, current_n_remove_vec, model_list, M, useGS, options);
                Temp_AICc(:,:,run_idx)   = aicc; 
                Temp_AIC(:,:,run_idx)    = aic; 
                Temp_RSE(:,:,run_idx)    = rse; 
                Temp_DW(:,:,run_idx)     = dw;
                Temp_MAPE(:,:,run_idx)   = mape;
                Temp_Params(:,:,:,run_idx) = prm;
            end
        else
            for run_idx = 1:N_RUNS
                [aicc, aic, rse, dw, mape, prm] = run_mc_step(L, all_t_data, all_Ndata_data, time_intervals, ...
                                               do_subsampling, current_n_remove_vec, model_list, M, useGS, options);
                Temp_AICc(:,:,run_idx)   = aicc; 
                Temp_AIC(:,:,run_idx)    = aic; 
                Temp_RSE(:,:,run_idx)    = rse; 
                Temp_DW(:,:,run_idx)     = dw;
                Temp_MAPE(:,:,run_idx)   = mape;
                Temp_Params(:,:,:,run_idx) = prm;
            end
        end
    end
    
    % --- AGGREGATION E PRINT LOG ---
    for i = 1:L 
        if isnan(current_n_remove_vec(i)), continue; end
        
        raw_aicc = squeeze(Temp_AICc(i, :, :));
        raw_aic  = squeeze(Temp_AIC(i, :, :));
        raw_rse  = squeeze(Temp_RSE(i, :, :));
        raw_dw   = squeeze(Temp_DW(i, :, :));
        raw_mape = squeeze(Temp_MAPE(i, :, :));
        
        valid_mask = ~any(isinf(raw_aicc) | isnan(raw_aicc), 1); 
        valid_runs = sum(valid_mask);
        
        if valid_runs > 0
            [~, win_idx_aicc] = min(raw_aicc(:, valid_mask), [], 1);
            [~, win_idx_aic]  = min(raw_aic(:, valid_mask), [], 1);
            [~, win_idx_rse]  = min(raw_rse(:, valid_mask), [], 1);
            for m = 1:M
                Persist_AICc(i, m, step_k) = sum(win_idx_aicc == m) / valid_runs * 100;
                Persist_AIC(i, m, step_k)  = sum(win_idx_aic == m) / valid_runs * 100;
                Persist_RSE(i, m, step_k)  = sum(win_idx_rse == m) / valid_runs * 100;
                Mean_Params(i, m, :, step_k) = mean(Temp_Params(i, m, :, valid_mask), 4, 'omitnan');
            end
            
            Mean_AICc(i,:,step_k)     = mean(raw_aicc(:, valid_mask), 2);
            Mean_AIC(i,:,step_k)      = mean(raw_aic(:, valid_mask), 2);
            Mean_RSE(i,:,step_k)      = mean(raw_rse(:, valid_mask), 2);
            Mean_DW(i,:,step_k)       = mean(raw_dw(:, valid_mask), 2);
            Mean_MAPE(i,:,step_k)     = mean(raw_mape(:, valid_mask), 2);
        else
            Persist_AICc(i,:,step_k) = 0; Persist_AIC(i,:,step_k)  = 0; Persist_RSE(i,:,step_k)  = 0; 
        end
        
        pct_rem = Actual_Pct(i, step_k) * 100;
        fprintf('Dataset: %-15s | Removed: %2d pts (%5.1f%%)\n', curve{i}, current_n_remove_vec(i), pct_rem);
        
        fprintf('  %-12s | %-8s | %-8s | %-8s || %-8s | %-8s | %-8s | %-6s | %-8s || %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n', ...
                'Model', 'Win AICc', 'Win AIC', 'Win RSE', 'm_AICc', 'm_AIC', 'm_RSE', 'm_DW', 'm_MAPE', 'lambda', 'K', 'N0', 'sigma', 'beta', 'lam*beta');
        fprintf('  --------------------------------------------------------------------------------------------------------------------------------------------------------\n');
        
        for m = 1:M
            p = squeeze(Mean_Params(i, m, :, step_k));
            sigma_val = p(4);
            if contains(model_list{m}, 'LogN', 'IgnoreCase', true)
                sigma_val = exp(sigma_val);
            end
            
            fprintf('  %-12s | %-7.1f%% | %-7.1f%% | %-7.1f%% || %-8.2f | %-8.2f | %-8.3f | %-6.2f | %-8.2f || %-8.4f | %-8.1f | %-8.1f | %-8.3f | %-8.4f | %-8.4f\n', ...
                model_list{m}, ...
                Persist_AICc(i,m,step_k), Persist_AIC(i,m,step_k), Persist_RSE(i,m,step_k), ...
                Mean_AICc(i,m,step_k), Mean_AIC(i,m,step_k), Mean_RSE(i,m,step_k), Mean_DW(i,m,step_k), Mean_MAPE(i,m,step_k), ...
                p(1), p(2), p(3), sigma_val, p(5), p(1)*p(5));
            
            % --- SALVATAGGIO RIGA CSV ---
            new_csv_row = {curve{i}, current_n_remove_vec(i), pct_rem, model_list{m}, ...
                           Persist_AICc(i,m,step_k), Persist_AIC(i,m,step_k), Persist_RSE(i,m,step_k), ...
                           Mean_AICc(i,m,step_k), Mean_AIC(i,m,step_k), Mean_RSE(i,m,step_k), ...
                           Mean_DW(i,m,step_k), Mean_MAPE(i,m,step_k), ...
                           p(1), p(2), p(3), sigma_val, p(5), p(1)*p(5)};
            csv_data = [csv_data; new_csv_row];
        end
        fprintf('\n');
    end
end 
elapsedTime = toc;
fprintf('\nAnalysis completed in %.2f minutes.\n', elapsedTime/60);

%% ===================== DATA EXPORT (MAT & CSV) ==========================
fprintf('\n--- Exporting Data ---\n');

% 1. Salva il file CSV
csv_filename = fullfile(save_dir, sprintf('Persistence_%s_Results_%d.csv', dataset_name, N_RUNS));
T_csv = cell2table(csv_data, 'VariableNames', csv_header);
writetable(T_csv, csv_filename);
fprintf('  > Saved CSV logs to: %s\n', csv_filename);

% 2. Salva il file MAT (Workspace)
mat_filename = fullfile(save_dir, sprintf('Persistence_%s_Workspace_%d.mat', dataset_name, N_RUNS));
save(mat_filename, 'Persist_AICc', 'Persist_AIC', 'Persist_RSE', ...
                   'Mean_AICc', 'Mean_AIC', 'Mean_RSE', 'Mean_DW', 'Mean_MAPE', ...
                   'Mean_Params', 'Actual_Pct', 'Removal_Plan', ...
                   'model_list', 'curve', 'data_paths', 'N_original_per_dataset', ...
                   't_start', 'N_RUNS', 'csv_data');
fprintf('  > Saved MATLAB variables to: %s\n', mat_filename);

%% ======================= PLOTTING 2x2 ===================================
fprintf('\n--- Generating Figures ---\n');
colors = lines(M);
styles = {'-o', '-s', '-^', '-d', '-v', '-p'}; 

for i = 1:L 
    valid_steps = ~isnan(Actual_Pct(i, :));
    x_vals = Actual_Pct(i, valid_steps) * 100;
    x_max = max(x_vals) + 5;
    
    hFig = figure('Name', sprintf('Metrics Evolution - %s', curve{i}), ...
                  'Position', [100, 100, 1000, 700]);
              
    subplot(2, 2, 1); hold on;
    y = squeeze(Persist_AICc(i, :, valid_steps)); box on;
    for m=1:M, plot(x_vals, y(m,:), styles{m}, 'LineWidth', 2, 'Color', colors(m,:)); end
    title('1. Winner Selection (AICc)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Win %'); xlabel('% Points Removed'); ylim([0 105]); xlim([0, x_max]); grid on;
    
    subplot(2, 2, 2); hold on;
    y = squeeze(Mean_DW(i, :, valid_steps));
    for m=1:M, plot(x_vals, y(m,:), styles{m}, 'LineWidth', 2, 'Color', colors(m,:)); end
    yline(2, 'r--', 'LineWidth', 1.5); box on;
    title('4. Mean Durbin-Watson', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('DW'); xlabel('% Points Removed'); ylim([0 3.5]); xlim([0, x_max]); grid on;
    
    subplot(2, 2, 3); hold on;
    y = squeeze(Mean_AICc(i, :, valid_steps));
    for m=1:M, plot(x_vals, y(m,:), styles{m}, 'LineWidth', 2, 'Color', colors(m,:)); end
    title('2. Mean AICc', 'FontSize', 12, 'FontWeight', 'bold'); box on;
    ylabel('AICc'); xlabel('% Points Removed'); xlim([0, x_max]); grid on;
    legend(model_list, 'Location', 'best');
    
    subplot(2, 2, 4); hold on;
    y = squeeze(Mean_RSE(i, :, valid_steps));
    for m=1:M, plot(x_vals, y(m,:), styles{m}, 'LineWidth', 2, 'Color', colors(m,:)); end
    title('3. Mean RSE', 'FontSize', 12, 'FontWeight', 'bold'); box on;
    ylabel('RSE'); xlabel('% Points Removed'); xlim([0, x_max]); grid on;

    % Salvataggio figure in .fig e .png
    base_figname = fullfile(save_dir, sprintf('Persistence_%s_%s_%d', dataset_name, curve{i}, N_RUNS));
    try 
        saveas(hFig, [base_figname, '.fig']); 
        saveas(hFig, [base_figname, '.png']);
    catch ME
        fprintf('  > Error saving figures for %s: %s\n', curve{i}, ME.message);
    end
end

diary off;
fprintf('Analysis complete. Results, figures and log correctly saved in %s.\n', save_dir);

%% ========================== HELPER FUNCTION =============================
function [t_aicc, t_aic, t_rse, t_dw, t_mape, t_params] = run_mc_step(L, all_t, all_N, t_int, subsamp, n_rem_vec, mods, M, GS, opts)
    t_aicc=nan(L,M); t_aic=nan(L,M); t_rse=nan(L,M); t_dw=nan(L,M); t_params=nan(L,M,5); t_mape=nan(L,M);
    for i = 1:L
        nr = n_rem_vec(i);
        if isnan(nr), continue; end
        
        t = all_t{i}; Ndata = all_N{i};
        
        if ~isempty(t_int{i})
            if length(t_int{i})==1, te=max(t); else, te=t_int{i}(2); end
            id = t>=t_int{i}(1) & t<=te; 
            t=t(id); Ndata=Ndata(id);
        end
        
        if subsamp && nr > 0
            [t, Ndata] = subsample_data_full_random(t, Ndata, nr); 
        end
        
        n_p = length(t);
        if n_p < 7, continue; end % Interrompe se i punti sono meno di 7
        
        for j = 1:M
            md = mods{j};
            try
                [l0, K0, N00, s0, b0, ~, lb, ub] = initialise_param(md, Ndata);
                
                if GS
                    [mle, nLL] = fit_growth_model_global(md,t,Ndata,[l0,K0,N00,s0,b0],lb,ub,[],opts);
                else
                    [mle, nLL] = fit_growth_model_local(md,t,Ndata,[l0,K0,N00,s0,b0],lb,ub,[],opts); 
                end
                
                [k_val, ssr, aic, ~, res, mape] = evaluate_fit(md, mle, t, Ndata, nLL);
                
                if n_p > k_val + 1
                    aicc = aic + (2*k_val*(k_val+1))/(n_p - k_val - 1); 
                else
                    aicc = Inf; 
                end
                
                if n_p > k_val
                    s_rse = sqrt(ssr / (n_p - k_val));
                else
                    s_rse = NaN;
                end
                
                dw = durbinWatsonTest(res(:));
                
                t_aicc(i,j) = aicc; 
                t_aic(i,j)  = aic; 
                t_rse(i,j)  = s_rse; 
                t_dw(i,j)   = dw;
                t_mape(i,j) = mape;
                
                curr_prm = nan(1,5);
                curr_prm(1:length(mle)) = mle;
                t_params(i,j,:) = curr_prm;
                
            catch
            end
        end
    end
end