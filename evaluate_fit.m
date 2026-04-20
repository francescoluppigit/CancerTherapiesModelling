function [k, SSR, AIC, Nmle, res, MAPE] = evaluate_fit(model_type, mle, t, Ndata, nLL)
    lambda = mle(1); K = mle(2); N0 = mle(3); 
    sigma0 = mle(4); beta0 = mle(5);
    md = lower(model_type);
    
    % Determine effective number of parameters (k)
    switch lower(md)
        case {'logistic', 'logisticlogn'}
            Nmle = K ./ (1 + ((K - N0)/N0) * exp(-lambda * t));
            k = 4; % lambda, K, N0, sigma
        case {'gompertz', 'gompertzlogn'}
            Nmle = K .* exp(log(N0/K) .* exp(-lambda * t));
            k = 4; % lambda, K, N0, sigma
        case {'richards', 'richardslogn'}
            Nmle = K*N0./(N0^beta0+(K^beta0-N0^beta0).*exp(-1*beta0*lambda*t)).^(1/beta0);
            k = 5; % lambda, K, N0, sigma, beta
        case 'exponential'
            C = beta0; % beta0 is the constant C
            Nmle = C + N0 .* exp(lambda * t);
            k = 4; % lambda, N0, sigma, C
    end
    
    % --- 2. Residuals ---

    % Nota: SSR si calcola solitamente sui dati grezzi per coerenza
    % ma per il plot dei residui useremo la variabile 'res' definita sopra.
    res_linear = Ndata - Nmle;
    SSR = sum(res_linear.^2);
    
    % Calculate Mean Absolute Percentage Error (MAPE)
    MAPE = mean(abs(res_linear ./ Ndata)) * 100;
    % per pesare correttamenete normale vs log-normale

    % Per i modelli LOG, i residui "veri" sono la differenza dei logaritmi
    if contains(md, 'logn'), res = log(max(Ndata, 1e-9)) - log(max(Nmle, 1e-9));
    else, res = Ndata - Nmle;    
    end    
    
    % n = length(t);
    % R2_raw = 1 - SSR / sum((Ndata - mean(Ndata)).^2);
    % R2 = 1-(1-R2_raw)*(n-1)/(n-k-1);
    
    % --- 3. AIC Comparison Correction (CRUCIALE) ---
    % Se il modello è stato fittato sui logaritmi (Log-Normale), la nLL restituita
    % dall'ottimizzatore è sulla scala log. Per confrontarla con la nLL della 
    % Logistica normale (scala lineare), dobbiamo aggiungere la somma dei log(Dati).
    % Questo è il "Termine Jacobiano".

    nLL_comparable = nLL; % Default per distribuzione normale

    if contains(md, 'logn')
        % Stiamo usando un modello log-normale
        % Correzione Jacobiano: nLL_raw = nLL_log + sum(log(Ndata))

        % Protezione contro log(0) se ci sono zeri nei dati
        data_clean = Ndata; 
        data_clean(data_clean<=0) = 1e-9; 

        jacobian_term = sum(log(data_clean));
        nLL_comparable = nLL + jacobian_term;
    end

    % Calcolo AIC usando la nLL corretta
    AIC = 2*k + 2*nLL_comparable;
end

%% MODELLI ESCLUSI
% case 'polynomial'
        %     % Parametri: b3, b2, b1, b0 messi in [lambda, K, N0, beta0] per comodità
        %     b3 = mle(1); b2 = mle(2); b1 = mle(3); b0 = mle(5); % Mappatura arbitraria
        %     Nmle = b3.*t.^3 + b2.*t.^2 + b1.*t + b0;
        %     k = 5;
        % case 'vonbertalanffy'
        %     % Analytical solution for Von Bertalanffy
        %     % Formula: N(t) = K * ( 1 - (1 - (N0/K)^(1/3)) * exp(-gamma * t) )^3
        %     term1 = (N0/K)^(1/3);
        %     Nmle = K .* ( 1 - (1 - term1) .* exp(-lambda * t) ).^3;
        %     k = 4; % lambda(gamma), K, N0, sigma
        % case 'biexponential'
        %     alpha_decay = mle(1); % stored in lambda slot
        %     Vol_Sens    = mle(2); % stored in K slot
        %     beta_growth = mle(3); % stored in N0 slot
        %     Vol_Res     = mle(5); % stored in beta0 slot
        %     % Formula: A*exp(-alpha*t) + B*exp(beta*t)
        %     Nmle = Vol_Sens .* exp(-alpha_decay * t) + Vol_Res .* exp(beta_growth * t);
        %     k = 5; % alpha, A, beta, sigma, B
        % case 'powerlaw'
        %     gamma_exp = mle(1); % stored in lambda
        %     alpha_scale = mle(2); % stored in K
        %     N_start = mle(3);     % stored in N0
        %     % Formula: N(t) = N0 + alpha * t^gamma
        %     Nmle = N_start + alpha_scale .* (t .^ gamma_exp);
        %     k = 4; % gamma, alpha, N0, sigma