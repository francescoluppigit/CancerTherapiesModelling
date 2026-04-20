%% ODE Parameter Estimation and Forecasting
% Fits a 3-compartment ODE system (U, I, V) to experimental data.
% Grid search for parameters [xi, m]
% Optimizes parameters [xi, m] and initial conditions [U0, I0, V0].

clc; clear; close all;

%% 1. Configuration & Data Setup
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

data_folder22 = 'Dati3\neutropenia';     % 0.2 - 0.4
data_folder23 = 'Dati3\spontneous\1';    % 0.2 - 0.4
data_folder24 = 'Dati3\testosterone';    %
data_folder25 = 'Dati3\review';          % 

data_folder = data_folder25;
RES = false; % residuals plot

% Forecasting configuration
% pct_start = 0.2; pct_end   = 0.5;

pct_start = 0.1; 
pct_end   = 0.4;

extra_forecast_days = 1;

% Optimization settings
metric_type  = 'std';    % std log weight
dataset_type = 'swing'; % growth swing general
use_grid_search = true;

if contains(metric_type, 'growth'), gridN = 10; else, gridN = 20; end

% Initial guess
params.U0 = 0.4;
params.I0 = 0.01;   
params.V0 = 0.1;

try
    [file_paths, file_names, L] = load_data_set(data_folder);
catch ME
    error('Dataset loading failed: %s', ME.message);
end

% Setup bounds for [xi, m]
switch lower(dataset_type)
    case 'growth'
        params.xi = 0.1; params.m = 0.5;
        lb_params = [1e-4, 1e-4]; ub_params = [1, 20]; 
        xi_grid = linspace(0.001, 0.1, gridN); m_grid = linspace(3, 10, gridN);
    case 'swing'
        params.xi = 0.001; params.m = 0.5;
        lb_params = [1e-4, 1e-4]; ub_params = [0.6, 1]; 
        xi_grid = linspace(0.0001, 0.1, gridN); m_grid = linspace(0.2, 1, gridN);
    case 'general'
        params.xi = 0.01; params.m = 0.5;
        lb_params = [1e-4, 1e-4]; ub_params = [1, 10]; 
        xi_grid = linspace(0.0001, 1, gridN); m_grid = linspace(0.1, 10, 2*gridN);
    otherwise
        error('Unknown dataset type: %s', dataset_type);
end

% Setup bounds for Initial Conditions [U0, I0, V0]
% Allowing the solver to explore realistic concentrations
lb_ic = [1e-4, 1e-4, 1e-4]; 
ub_ic = [2.0,  1.0,  2.0];  

% Final Full Bounds: [xi, m, U0, I0, V0]
lb_full = [lb_params, lb_ic];
ub_full = [ub_params, ub_ic];

%% 2. Main Processing Loop
for j = 1:L
    fprintf('\n=================================================\n');
    fprintf('Processing %d/%d: %s\n', j, L, file_names{j});
    
    [raw_t, raw_y] = import_data(file_paths{j});
    N = length(raw_t);
    idx_start = max(1, floor(pct_start * N) + 1); 
    idx_end   = min(N, floor(pct_end * N));       
    
    if idx_end <= idx_start
        warning('Invalid percentage range for %s. Using full dataset.', file_names{j});
        idx_start = 1; idx_end = N;
    end
    
    t_train = raw_t(idx_start:idx_end);
    y_train = raw_y(idx_start:idx_end);
    
    scale_factor = max(y_train) - min(y_train);
    if scale_factor == 0, scale_factor = 1; end
    y_train_norm = y_train / scale_factor;

    % --- 2D Grid Search for xi and m (keeping U0, I0, V0 fixed at nominal) ---
    best_guess_2d = [params.xi, params.m];
    if use_grid_search
        disp('  > Running 2D Grid Search (xi, m)...');
        best_ssr = inf;
        for p_xi = xi_grid
            for p_m = m_grid
                % Temporarily test with 5 parameters, where ICs are fixed to nominal
                p_test_5d = [p_xi, p_m, params.U0, params.I0, params.V0];
                res = compute_residuals(p_test_5d, params, y_train_norm, t_train, metric_type);
                ssr = sum(res.^2);
                if isfinite(ssr) && (ssr < best_ssr)
                    best_ssr = ssr;
                    best_guess_2d = [p_xi, p_m];
                end
            end
        end
        fprintf('  > 2D Best guess: xi = %.4f, m = %.4f\n', best_guess_2d(1), best_guess_2d(2));
    end
    
    % --- Optimization (xi, m, U0, I0, V0) ---
    disp('  > Optimizing all 5 parameters (including Initial Conditions)...');
    best_guess_5d = [best_guess_2d(1), best_guess_2d(2), params.U0, params.I0, params.V0];
    
    [p_opt, resnorm, residual] = optimize_parameters(best_guess_5d, params, y_train_norm, t_train, lb_full, ub_full, metric_type);
    
    % Save fitted parameters
    params_fit = params;
    params_fit.xi = p_opt(1);
    params_fit.m  = p_opt(2);
    params_fit.U0 = p_opt(3);
    params_fit.I0 = p_opt(4);
    params_fit.V0 = p_opt(5);
    
    % --- Forecasting ---
    t_forecast_max = max(raw_t) + extra_forecast_days;
    sol_forecast = solve_ode(params_fit, t_forecast_max);
    
    U_rescaled = sol_forecast.y(1,:) .* scale_factor;
    I_rescaled = sol_forecast.y(2,:) .* scale_factor;
    V_rescaled = sol_forecast.y(3,:) .* scale_factor;
    forecast_total_volume = U_rescaled + I_rescaled;
    
    %% --- Plotting (2x1 Layout) ---
    fig = figure(j);
    set(fig, 'Name', ['Dynamics & Fit: ' file_names{j}], 'NumberTitle', 'off', 'Position', [100, 100, 1420, 800]);
    
    % Subplot 1: Solutions (Top)
    subplot(2, 1, 1); hold on; grid on;
    plot(sol_forecast.x, U_rescaled, 'LineWidth', 2.5, 'DisplayName', 'U');
    plot(sol_forecast.x, I_rescaled, 'LineWidth', 2.5, 'DisplayName', 'I');
    plot(sol_forecast.x, V_rescaled, 'LineWidth', 2.5, 'DisplayName', 'V');
    title('Solutions');
    ylabel('Volume');
    xlim([0, t_forecast_max]);
    legend('Location', 'best');
    set(gca, 'FontSize', 20); 
    
    % Subplot 2: Total Volume Fit and Forecast (Bottom)
    subplot(2, 1, 2); hold on; grid on;
    plot(sol_forecast.x, forecast_total_volume, 'm-', 'LineWidth', 3, 'DisplayName', 'U+I');
    plot(t_train, y_train, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 8, 'DisplayName', 'Training');
    
    idx_excluded = [1:(idx_start-1), (idx_end+1):N];
    if ~isempty(idx_excluded)
        plot(raw_t(idx_excluded), raw_y(idx_excluded), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 8, 'DisplayName', 'Testing');
    end
    
    xline(raw_t(idx_start), '--k', 'Fit Start', 'HandleVisibility', 'off', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 2, 'FontSize', 16);
    xline(raw_t(idx_end), '--k', 'Fit End', 'HandleVisibility', 'off', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 2, 'FontSize', 16);
    
    title(sprintf('U+I Fit: \\xi=%-0.4f, m=%-0.4f', p_opt(1), p_opt(2)));
    xlabel('Time (days)'); ylabel('Total Volume');
    ylim([0, max([forecast_total_volume(:); raw_y(:)]) * 1.1]);
    xlim([0, t_forecast_max]);
    legend('Location', 'best');
    set(gca, 'FontSize', 20); 
    drawnow;

    %% --- Plotting Residuals (Train + Test) ---
    if RES
        fig_res = figure(j + L); 
        set(fig_res, 'Name', ['Residuals: ' file_names{j}], 'NumberTitle', 'off', 'Position', [800, 200, 700, 400]);
        hold on; grid on;
    
        % 1. TRAIN RESIDUALS
        y_interp_train = deval(sol_forecast, t_train);
        model_train_rescaled = (y_interp_train(1,:)' + y_interp_train(2,:)') .* scale_factor;
        abs_residuals_train = y_train(:) - model_train_rescaled(:);
    
        % Disegna i residui di training (Blu)
        stem(t_train, abs_residuals_train, 'Color', 'b', 'MarkerFaceColor', 'b', ...
             'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Train Error');
    
        max_res_train = max(abs(abs_residuals_train));
        max_res_test = 0; % Inizializzazione sicura
    
        % 2. TEST RESIDUALS (Se esistono dati esclusi)
        if ~isempty(idx_excluded)
            t_test = raw_t(idx_excluded);
            y_test = raw_y(idx_excluded);
    
            y_interp_test = deval(sol_forecast, t_test);
            model_test_rescaled = (y_interp_test(1,:)' + y_interp_test(2,:)') .* scale_factor;
            abs_residuals_test = y_test(:) - model_test_rescaled(:);
    
            % Disegna i residui di testing (Nero)
            stem(t_test, abs_residuals_test, 'Color', 'k', 'MarkerFaceColor', 'k', ...
                 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Test Error');
    
            max_res_test = max(abs(abs_residuals_test));
    
            % Aggiungi linee verticali tratteggiate per delineare la zona di training
            xline(raw_t(idx_start), '--k', 'HandleVisibility', 'off', 'LineWidth', 1.5);
            xline(raw_t(idx_end), '--k', 'HandleVisibility', 'off', 'LineWidth', 1.5);
        end
    
        % Linea dello zero
        yline(0, 'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off'); 
    
        % Fascia di tolleranza visiva +/- 10
        yline(10, '--g', 'HandleVisibility', 'off', 'Alpha', 0.5);
        yline(-10, '--g', 'HandleVisibility', 'off', 'Alpha', 0.5);

        % Fascia di tolleranza visiva +/- 10
        yline(50, '--r', 'HandleVisibility', 'off', 'Alpha', 0.5);
        yline(-50, '--r', 'HandleVisibility', 'off', 'Alpha', 0.5);
    
        title(sprintf('Residual Analysis: %s', file_names{j}));
        xlabel('Time (days)'); 
        ylabel('Error');
        legend('Location', 'best');
        set(gca, 'FontSize', 16);
    
        % Adatta l'asse Y considerando sia gli errori di train che di test
        max_total_res = max(max_res_train, max_res_test);
        if max_total_res > 0
            ylim([-max_total_res * 1.2, max_total_res * 1.2]);
        end
        drawnow;
    end

    %% --- Evaluation Metrics ---
    n_points = length(t_train);
    ssr_val  = sum((residual * scale_factor).^2);
    rmse_val = sqrt(ssr_val / n_points);
    std_val  = std(y_train_norm) * scale_factor; 
    
    nrmse_pct = (rmse_val / scale_factor) * 100;
    rsr_val   = rmse_val / std_val; 
    
    fprintf('  > Metrics:\n');
    fprintf('    RMSE:  %.4f\n', rmse_val);
    fprintf('    nRMSE: %.2f%%\n', nrmse_pct);
    fprintf('    RSR:   %.2f\n', rsr_val);
    fprintf('  Params : xi = %.4f | m = %.4f | U0 = %.3f | I0 = %.3f | V0 = %.3f\n', ...
          params_fit.xi, params_fit.m, params_fit.U0, params_fit.I0, params_fit.V0);
end
disp('Analysis complete.');

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================
function [p_opt, resnorm, residual] = optimize_parameters(p_guess, params, y_data, t_data, lb, ub, metric)
    options = optimoptions(@lsqnonlin, ...
        'Algorithm', 'trust-region-reflective', ...
        'Display', 'off', ... 
        'MaxIterations', 200, ...      % Increased max iter for 5 variables
        'StepTolerance', 1e-6, ...
        'FunctionTolerance', 1e-6);
    
    obj_fun = @(p) compute_residuals(p, params, y_data, t_data, metric);
    [p_opt, resnorm, residual] = lsqnonlin(obj_fun, p_guess, lb, ub, options);
end

function val = compute_residuals(p_opt, params_fixed, y_data, t_data, metric)
    % p_opt is now expected to be a 1x5 vector: [xi, m, U0, I0, V0]
    high_penalty = ones(length(y_data), 1) * 1e6;
    T_max = max(t_data);

    if any(~isfinite(p_opt)) || any(p_opt < 0)
        val = high_penalty; return;
    end

    % Update all 5 parameters for the current iteration
    p_curr = params_fixed; 
    p_curr.xi = p_opt(1);
    p_curr.m  = p_opt(2);
    p_curr.U0 = p_opt(3);
    p_curr.I0 = p_opt(4);
    p_curr.V0 = p_opt(5);
    
    try
        sol = solve_ode(p_curr, T_max);
    catch 
        val = high_penalty; return;
    end
    
    if isempty(sol.x) || length(sol.x) < 2 || sol.x(end) < (T_max - 1e-3) 
        val = high_penalty; return;
    end
    
    try
        y_interp = deval(sol, t_data);
        model_volume = y_interp(1, :)' + y_interp(2, :)'; 
        data_col = y_data(:);
        
        switch lower(metric)
            case 'std'
                val = model_volume - data_col;
            case 'weights'
                weights = linspace(1, 5, length(data_col))';
                val = (model_volume - data_col) .* weights;
            case 'log'
                eps = 1e-3;
                val = log(model_volume + eps) - log(data_col + eps);
            otherwise
                error('Unknown metric: %s', metric);
        end
        val(isnan(val)) = 1e6; 
    catch
        val = high_penalty;
    end
end

function sol = solve_ode(params, t_max)
    % Extracts the dynamically updated initial conditions
    init_vals = [params.U0, params.I0, params.V0];
    
    % Consider switching to ode15s if the system becomes too stiff
    options = odeset('RelTol', 1e-3, 'AbsTol', 1e-6, 'NonNegative', [1 2 3]);
    ode_fun = @(t, y) system_equations(t, y, params.xi, params.m);
    
    sol = ode45(ode_fun, [0, t_max], init_vals, options);
end

function dydt = system_equations(~, y, xi, m)
    U = y(1);
    I = y(2);
    V = y(3);
    
    dU = xi*U - U*V;
    dI = U*V - I;
    dV = -m*V + I; 
    
    dydt = [dU; dI; dV];
end