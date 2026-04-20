function [e] = likelihood_Exponential(lambda, N0, sigma, C, t, Ndata)
    
    % Robustness check
    if sigma <= 1e-10 || N0 <= 0
        e = -1e100; % Penality
        return;
    end
    
    N = C + N0 .* exp(lambda * t);
    
    % Check for for NaN o Inf
    if any(isnan(N)) || any(isinf(N)) || ~isreal(N)
        e = -1e100;
        return;
    end
    
    % log-likelihood (Gaussian error distribution)
    y = log(normpdf(N, Ndata, sigma));
    
    % Cases of log(0) = -Inf
    y(y == -Inf) = -1e100;
    e = sum(y);
    
    % Final check
    if isnan(e) || isinf(e)
        e = -1e100;
    end
end
