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

data_folder = data_folder1; 
[data_paths, curve, L] = load_data_set(data_folder); 

%% ======================= MONTE CARLO CONFIG =============================
rng(42); % Fix seed for reproducibility
N_RUNS = 500; 
percentage_to_remove = 0.15;

%% ============================= INTERVALS & MODELS =======================
time_intervals = cell(L, 1);
t_start = 11;
for i = 1:L, time_intervals{i} = [t_start]; end

% MODELS
model_list1 = {'Logistic', 'Gompertz', 'Richards'};
model_list2 = {'Logistic', 'LogisticLog1'};
model_list = model_list1;
M = length(model_list); 
useGS = false; 

%% ======================= DATA PRE-LOADING ===============================
fprintf('Pre-loading %d datasets...\n', L);
all_t_data = cell(L, 1);
all_Ndata_data = cell(L, 1);
n_tot_points = 0;

for i = 1:L
    [all_t_data{i}, all_Ndata_data{i}] = import_data(data_paths{i});
    if length(all_t_data{i}) > n_tot_points
        n_tot_points = length(all_t_data{i});
    end
end

%% ======================= MONTE CARLO POINTS =============================
n_points_to_remove = round(percentage_to_remove * n_tot_points); 
useParallel = true; 

options = optimoptions('fmincon','Display','off',...
    'MaxFunctionEvaluations',5000,'OptimalityTolerance',1e-10);

%% --- Logging Setup ---
[~, folder_name_base, ~] = fileparts(pwd); 
log_filename = sprintf('MC_Log_%s.txt', datestr(now, 'yyyymmdd_HHMM'));
diary(log_filename);
fprintf('--- Starting Monte Carlo Analysis ---\n');
fprintf('Date/Time: %s\n\n', datestr(now));

%% ========================= STORAGE INITIALIZATION =======================
AICc_Win    = zeros(L, N_RUNS);           % Winning AICc
AICc_All    = zeros(L, M, N_RUNS);        % All AICc
RES_Win     = zeros(L, N_RUNS);           % Winning RES
DW_Win      = zeros(L, N_RUNS);           % Winning DW
Model_Win   = cell(L, N_RUNS);            % Winning model name
Params_Win  = cell(L, N_RUNS);            % Winning parameters

if useParallel
    p = gcp('nocreate');
    if isempty(p), try, parpool('Processes'); catch, parpool; end; end
end

%% ======================= MONTE CARLO LOOP ===============================
fprintf('\nStarting Monte Carlo simulation (%d runs, removing %d points)...\n', N_RUNS, n_points_to_remove);
tic; 

if useParallel
    parfor run_idx = 1:N_RUNS
        [AICc_Win(:, run_idx), AICc_All(:,:,run_idx), RES_Win(:, run_idx), ...
         DW_Win(:, run_idx), Model_Win(:, run_idx), Params_Win(:, run_idx)] = ...
            run_mc_iteration(L, all_t_data, all_Ndata_data, time_intervals, ...
                             n_points_to_remove, model_list, M, useGS, options);
    end
else 
    for run_idx = 1:N_RUNS 
        if mod(run_idx, 50) == 0, fprintf('  Run %d/%d\n', run_idx, N_RUNS); end
        [AICc_Win(:, run_idx), AICc_All(:,:,run_idx), RES_Win(:, run_idx), ...
         DW_Win(:, run_idx), Model_Win(:, run_idx), Params_Win(:, run_idx)] = ...
            run_mc_iteration(L, all_t_data, all_Ndata_data, time_intervals, ...
                             n_points_to_remove, model_list, M, useGS, options);
    end
end

elapsedTime = toc; 
fprintf('Monte Carlo simulation completed in %.2f minutes.\n', elapsedTime/60);

%% ======================= ANALYSIS & PLOTTING ============================
fprintf('\n--- Monte Carlo Results Analysis ---\n');
[~, folder_name_base, ~] = fileparts(data_folder);
if isempty(folder_name_base), [~, folder_name_base, ~] = fileparts(data_folder(1:end-1)); end
perc_str = sprintf('%d', round(percentage_to_remove * 100));

for i = 1:L 
    dataset_model_selections = Model_Win(i, :);
    valid_runs = sum(~strcmp(dataset_model_selections, 'Failed') & ~strcmp(dataset_model_selections, 'NoData'));
    
    % --- 1. Consolidated Report Table ---
    fprintf('\n====================================================================================================================\n');
    fprintf('Dataset: %-15s | Runs: %d | Valid: %d\n', curve{i}, N_RUNS, valid_runs);
    fprintf('====================================================================================================================\n');
    fprintf('%-15s | %-6s | %-9s | %-8s | %-7s || %-8s | %-8s | %-8s | %-8s | %-8s\n', ...
            'Model', 'Win %', 'Mean AICc', 'Mean RES', 'Mean DW', 'Lambda', 'K', 'N0', 'Sigma', 'Beta');
    fprintf('--------------------------------------------------------------------------------------------------------------------\n');
    
    for m_idx = 1:M
        mdl = model_list{m_idx};
        idx_won = strcmp(dataset_model_selections, mdl);
        win_count = sum(idx_won);
        win_perc = 0; if valid_runs > 0, win_perc = (win_count / valid_runs) * 100; end
        
        if win_count > 0
            % Compute mean metrics for winning runs
            mean_aicc = mean(AICc_Win(i, idx_won), 'omitnan');
            mean_res  = mean(RES_Win(i, idx_won), 'omitnan');
            mean_dw   = mean(DW_Win(i, idx_won), 'omitnan');
            
            % Extract and average parameters safely
            valid_params = cellfun(@(x) x(:)', Params_Win(i, idx_won), 'UniformOutput', false);
            p_mat = vertcat(valid_params{:}); 
            
            if ~isempty(p_mat), mean_p = mean(p_mat, 1, 'omitnan'); else, mean_p = nan(1, 5); end
            if length(mean_p) < 5, mean_p(5) = NaN; end % Pad if < 5 params
            
            b_str = '-'; if strcmp(mdl, 'Richards') || ~isnan(mean_p(5)), b_str = sprintf('%.4f', mean_p(5)); end
            
            fprintf('%-15s | %5.1f%% | %9.2f | %8.3f | %7.2f || %8.4f | %8.1f | %8.1f | %8.3f | %8s\n', ...
                    mdl, win_perc, mean_aicc, mean_res, mean_dw, ...
                    mean_p(1), mean_p(2), mean_p(3), mean_p(4), b_str);
        else
            fprintf('%-15s | %5.1f%% | %9s | %8s | %7s || %8s | %8s | %8s | %8s | %8s\n', ...
                    mdl, 0, '-', '-', '-', '-', '-', '-', '-', '-');
        end
    end
    fprintf('====================================================================================================================\n');
    
    % --- 2. Build Parameter Matrix for Plotting ---
    winning_params_matrix = nan(N_RUNS, 5); 
    for run_idx = 1:N_RUNS
        params = Params_Win{i, run_idx};
        if ~isempty(params)
            len = length(params);
            winning_params_matrix(run_idx, 1:len) = params;
        end
    end
    
    % --- 3. PLOTTING (Grid 2x3) ---
    hFig = figure('Name', sprintf('MC Analysis - %s', curve{i}), 'Position', [100, 100, 1420, 800]);
    
    % Plot 1: Model Selection
    subplot(2, 3, 1); hold on; box on;
    [unique_models, ~, idx] = unique(dataset_model_selections);
    valid_model_idx = ~contains(unique_models, {'Failed', 'NoData'});
    counts = accumarray(idx, 1);
    percentages = 0; if valid_runs > 0, percentages = (counts / valid_runs) * 100; end
    
    if any(valid_model_idx)
        bar(categorical(unique_models(valid_model_idx)), percentages(valid_model_idx));
    end
    title('1. Winner Selection (%)'); ylabel('%'); ylim([0 100]); grid on;
    
    % Plot 2: Durbin-Watson Distribution
    subplot(2, 3, 2); hold on; box on;
    histogram(DW_Win(i, :), 30, 'FaceColor', [0.4660 0.6740 0.1880]);
    xline(2, 'r--', 'Optimal DW=2', 'LineWidth', 2);
    title('2. Durbin-Watson Stat'); xlabel('DW Value'); ylabel('Count'); grid on;
    
    % Plot 3: Lambda (Param 1)
    subplot(2, 3, 3); 
    histogram(winning_params_matrix(:, 1), 30); box on;
    title('3. \lambda Distribution'); xlabel('\lambda'); grid on;
    
    % Plot 4: K (Param 2)
    subplot(2, 3, 4);
    histogram(winning_params_matrix(:, 2), 30); box on;
    title('4. K Distribution'); xlabel('K'); grid on;
    
    % Plot 5: Sigma (Param 4)
    subplot(2, 3, 5);
    histogram(winning_params_matrix(:, 4), 30); box on;
    title('5. \sigma_{mle} Distribution'); xlabel('\sigma'); grid on;    
    
    % Plot 6: Beta (Param 5 - Only if Richards wins)
    subplot(2, 3, 6);
    beta_vals = winning_params_matrix(:, 5); box on;
    is_richards = strcmp(dataset_model_selections, 'Richards')';
    beta_vals(~is_richards) = NaN; % Filter: keep beta only when Richards wins
    
    if any(~isnan(beta_vals))
        histogram(beta_vals(~isnan(beta_vals)), 30, 'FaceColor', [0.8500 0.3250 0.0980]);
        title('6. \beta (Richards Only)'); xlabel('\beta'); grid on;
    else
        title('6. \beta (Not Applicable)');
        set(gca, 'XTick', [], 'YTick', []);
        text(0.5, 0.5, 'Richards not selected', 'HorizontalAlignment', 'center', 'FontSize', 18);
    end
    
    % Apply global font size to all subplots
    set(findall(hFig,'-property','FontSize'),'FontSize', 18);
    
    % Save figure
    fig_name = sprintf('MC_%s_%s_Rem%s', folder_name_base, curve{i}, perc_str);
    try 
        saveas(hFig, [fig_name '.fig']); 
        saveas(hFig, [fig_name '.png']); % Auto-save as PNG
    catch
    end
end 

diary off; 
fprintf('\nAnalysis complete. Log saved.\n');

%% ========================== HELPER FUNCTION =============================
function [best_aicc, all_aicc, best_rse, best_dw, best_model, best_params] = ...
    run_mc_iteration(L, all_t, all_N, t_int, n_rem, mods, M, GS, opts)
    
    best_aicc   = nan(L, 1);
    all_aicc    = nan(L, M);
    best_rse    = nan(L, 1);
    best_dw     = nan(L, 1);
    best_model  = repmat({'Failed'}, L, 1);
    best_params = cell(L, 1);
    
    for i = 1:L
        t = all_t{i}; Ndata = all_N{i};
        
        % Time interval filter
        if ~isempty(t_int{i})
            if length(t_int{i})==1, te=max(t); else, te=t_int{i}(2); end
            id = t>=t_int{i}(1) & t<=te; 
            t=t(id); Ndata=Ndata(id);
        end
        
        % Random subsampling
        if n_rem > 0
            [t, Ndata] = subsample_data_full_random(t, Ndata, n_rem);
        end
        
        n_p = length(t);
        if n_p < 5
            best_model{i} = 'NoData';
            continue; 
        end
        
        % Temp vars to find the best model
        curr_best_aicc = Inf;
        curr_best_rse  = NaN;
        curr_best_dw   = NaN;
        curr_best_mod  = 'Failed';
        curr_best_par  = [];
        
        for j = 1:M
            md = mods{j};
            try
                [l0, K0, N00, s0, b0, ~, lb, ub] = initialise_param(md, Ndata);
                
                if GS
                    [mle, nLL] = fit_growth_model_global(md,t,Ndata,[l0,K0,N00,s0,b0],lb,ub,[],opts);
                else
                    [mle, nLL] = fit_growth_model_local(md,t,Ndata,[l0,K0,N00,s0,b0],lb,ub,[],opts); 
                end
                
                [k_val, ssr, aic, ~, res, ~] = evaluate_fit(md, mle, t, Ndata, nLL);
                
                % 1. Compute AICc
                if n_p > k_val + 1
                    aicc_val = aic + (2*k_val*(k_val+1))/(n_p - k_val - 1); 
                else
                    aicc_val = Inf; 
                end
                all_aicc(i, j) = aicc_val;
                
                % Update winner
                if aicc_val < curr_best_aicc
                    curr_best_aicc = aicc_val;
                    curr_best_mod  = md;
                    curr_best_par  = mle;
                    curr_best_dw   = durbinWatsonTest(res(:));
                    
                    if n_p > k_val
                        curr_best_rse = sqrt(ssr / (n_p - k_val));
                    else
                        curr_best_rse = NaN;
                    end
                end
            catch
                % Failed fit leaves NaN in all_aicc(i,j)
            end
        end 
        
        % Save winner for dataset i
        if ~isinf(curr_best_aicc)
            best_aicc(i)   = curr_best_aicc;
            best_rse(i)    = curr_best_rse;
            best_dw(i)     = curr_best_dw;
            best_model{i}  = curr_best_mod;
            best_params{i} = curr_best_par;
        end
    end
end