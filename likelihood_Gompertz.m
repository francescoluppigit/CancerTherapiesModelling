function [g] = likelihood_Gompertz(lambda, K, N0, sigma, t,Ndata)
    
    % Robustness check per parametri non validi
    % Check N0 >= K since log(N0/K) is not defined for N0 > K (negative
    % growth)
    if sigma <= 1e-10 || N0 <= 0 || K <= 0 || N0 >= K
        g = -1e100; % Penality
        return;
    end
    
    N =  K*exp(log(N0/K)*exp(-1*lambda*t'));
    
    % Check for for NaN o Inf
    if any(isnan(N)) || any(isinf(N)) || ~isreal(N)
        g = -1e100;
        return;
    end
    
    y = log(normpdf(N',Ndata,sigma)); 
    y(y == -Inf) = -1e100; 
    g = sum(y);
    
    % Final check
    if isnan(g) || isinf(g)
        g = -1e100;
    end
end
