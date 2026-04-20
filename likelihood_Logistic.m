function [l] = likelihood_Logistic(lambda, K, N0, sigma, t, Ndata)
    
    % Robustness check
    if sigma <= 1e-10 || N0 <= 0 || K <= 0
        l = -1e100; % Penality
        return;
    end
    
    N = K*N0./(N0+(K-N0).*exp(-1*lambda*t'));
    
    % Check for NaN o Inf (e.g. N0 > K)
    if any(isnan(N)) || any(isinf(N)) || ~isreal(N)
        l = -1e100;
        return;
    end

    y = log(normpdf(N',Ndata,sigma)); 
    y(y == -Inf) = -1e100; 
    l = sum(y);
    
    % Final check
    if isnan(l) || isinf(l)
        l = -1e100;
    end
end
