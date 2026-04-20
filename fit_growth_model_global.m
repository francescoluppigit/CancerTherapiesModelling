function [mle, nLL] = fit_growth_model_global(model_type, t, Ndata, initParams, lb, ub, nonlcon, options)
    
    % Use the shared wrapper function
    funmle = @(p) likelihood_wrapper(model_type, p, t, Ndata);
    
    % --- BEGIN GLOBAL RESEARCH ---

    % 1. Def optim problem for fmincon (local solver)
    problem = createOptimProblem('fmincon', ...
        'x0', initParams, ... 
        'objective', funmle, ...
        'lb', lb, 'ub', ub, ...
        'nonlcon', nonlcon, ...
        'options', options);

    % 2.a MultiStart
    % ms = MultiStart('FunctionTolerance', 1e-6);
    % num_starts = 50; % N random start points
    % [mle, nLL_value] = run(ms, problem, num_starts);

    % 2.b GlobalSearch
    gs = GlobalSearch('FunctionTolerance', 1e-6);
    [mle, nLL_value] = run(gs, problem);

    nLL = -nLL_value; % MLE minimise -LL

    % --- END GLOBAL RESEARCH ---
end