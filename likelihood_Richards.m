function [r] = likelihood_Richards(lambda, K, N0, sigma, beta, t,Ndata)
 
    % Robustness check
    if sigma <= 1e-10 || N0 <= 0 || K <= 0 || beta <= 0
        r = -1e100; % Penality
        return;
    end
    
    base_expr = (N0^beta+(K^beta-N0^beta).*exp(-1*beta*lambda*t'));
    
    if any(base_expr <= 0) || ~isreal(base_expr)
         r = -1e100;
         return;
    end
 
    N = K*N0./(base_expr).^(1/beta);
    
    % Check for NaN o Inf
    if any(isnan(N)) || any(isinf(N)) || ~isreal(N)
        r = -1e100;
        return;
    end
    
    y = log(normpdf(N',Ndata,sigma)); 
    y(y == -Inf) = -1e100;
    r = sum(y);
    
    % Final check
    if isnan(r) || isinf(r)
        r = -1e100;
    end
end