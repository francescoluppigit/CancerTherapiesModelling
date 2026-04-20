function [mle, nLL] = fit_growth_model_local(model_type, t, Ndata, initParams, lb, ub, nonlcon, options)
    
    % Use the shared wrapper function
    funmle = @(p) likelihood_wrapper(model_type, p, t, Ndata);
    
    % --- BEGIN LOCAL RESEARCH ---

    mle = fmincon(funmle, initParams, [], [], [], [], lb, ub, nonlcon, options);
    nLL = funmle(mle); % last -LL value
    
    % --- END LOCAL RESEARCH ---
end