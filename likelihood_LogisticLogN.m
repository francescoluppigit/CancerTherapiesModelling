function [l] = likelihood_LogisticLogN(lambda, K, N0, sigma, t, Ndata)
    % Restituisce la LOG-LIKELIHOOD (LL) non corretta (nello spazio logaritmico).
    % Lo Jacobiano verrà aggiunto successivamente in evaluate_fit.m
    
    penalty = -1e20; 

    % Controllo parametri 
    if sigma <= 1e-4 || N0 <= 1e-3 || K <= 1e-3
        l = penalty;
        return;
    end
    
    % Modello Logistico Standard
    N = K*N0./(N0+(K-N0).*exp(-lambda*t'));
    
    % Protezione contro valori non positivi 
    if any(N <= 0) || any(isnan(N)) || any(isinf(N))
        l = penalty;
        return;
    end
    
    % Calcolo della Log-Likelihood nello spazio logaritmico (senza Jacobiano)
    n = length(Ndata);
    log_res = log(Ndata) - log(N');
    
    l = - (n/2)*log(2*pi) - n*log(sigma) - sum(log_res.^2)/(2*sigma^2);
    
    % Controllo finale
    if isnan(l) || isinf(l)
        l = penalty;
    end
end