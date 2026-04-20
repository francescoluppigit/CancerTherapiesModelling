function [lambda0, K0, N00, sigma0, beta0, nonlcon, lb, ub] = initialise_param(model_type, Ndata)
    beta0  = 0; nonlcon=[];
    molt = 2; % used for aplifying carrying capacity K
    if contains(model_type, 'logn'), sigma0 = 0.1; sigmalb = 1e-2; sigmaub = 2;
    else, sigma0 = std(Ndata)/2; sigmalb = 1e-3; sigmaub = std(Ndata)*10; end

    switch lower(model_type)    % lb, ub: [lambda, K, N0, sigma, beta]
        case {'logistic', 'logisticlogn'}
            lambda0 = 0.05; K0 = max(Ndata)*1.1; N00 = Ndata(1);
            lb = [1e-6, 0.5*max(Ndata), 0.5*min(Ndata), sigmalb, 0]; 
            ub = [1, molt*max(Ndata), max(Ndata), sigmaub, 0];
        case {'gompertz', 'gompertzlogn'}
            lambda0 = 0.001; K0 = max(Ndata)*1.1; N00 = Ndata(1);
            lb = [1e-6, 0.5*max(Ndata), 0.5*min(Ndata), sigmalb, 0]; 
            ub = [1, molt*max(Ndata), max(Ndata), sigmaub, 0];
        case {'richards', 'richardslogn'}
            lambda0 = 0.001; K0 = max(Ndata)*1.1; N00 = Ndata(1); beta0  = 1;
            lb = [1e-6, 0.5*max(Ndata), 0.5*min(Ndata), sigmalb, 0.01]; 
            ub = [1, molt*max(Ndata), max(Ndata), sigmaub, 10];
        case 'exponential'
            lambda0 = 0.01;             % growth rate
            K0 = 0;                     % DUMMY
            N00 = Ndata(1);             % initial value
            % beta0 is the vertical shift
            beta0  = 0.0;               % Initial constant C0
            % lb: [lambda, K, N0, sigma, C]
            lb = [1e-6, 0, 0.5*min(Ndata), sigmalb, 0]; 
            ub = [1,    0, 2*max(Ndata), sigmaub, 0.5*min(Ndata)]; 
    end
end

%% MODELLI ESCLUSI
% case 'vonbertalanffy'
        %     % Mapping: lambda -> gamma, K -> K, N0 -> N0
        %     lambda0 = 0.1; 
        %     K0 = max(Ndata)*1.2; 
        %     N00 = Ndata(1); 
        %     sigma0 = std(Ndata)/2;
        %     beta0 = 0; % Unused
        %     % Bounds similar to Gompertz
        %     lb = [1e-6, 0.5*max(Ndata), 0.5*min(Ndata), 1e-3, 0]; 
        %     ub = [1, 5*max(Ndata), max(Ndata), std(Ndata)*10, 0];
        % case 'biexponential'
        %     % Fit intelligente: Alpha(decay) domina all'inizio, Beta(growth) alla fine
        %     % A ~ Ndata(1), B ~ piccolo
        %     lambda0 = 0.1;           % alpha (decay rate)
        %     K0 = max(Ndata(1), 0.1); % A (Volume Sensibile)
        %     N00 = 0.05;              % beta (growth rate)
        %     sigma0 = std(Ndata)/2;
        %     beta0 = max(min(Ndata)/2, 0.1); % B (Volume Resistente)
        %     % Bounds: A e B devono essere positivi
        %     lb = [0,    0, 0,    1e-3, 0]; 
        %     ub = [5, max(Ndata)*2, 2, std(Ndata)*10, max(Ndata)];
        % case 'polynomial'
        %     % Inizializzazione per Polinomio Cubico: ax^3 + bx^2 + cx + d
        %     % Usa polyfit per una stima iniziale robusta
        %     if length(t) >= 4
        %         p = polyfit(t, Ndata, 3);
        %     else
        %         p = [0, 0, 0, mean(Ndata)];
        %     end
        % 
        %     lambda0 = p(1); % Coeff x^3
        %     K0      = p(2); % Coeff x^2
        %     N00     = p(3); % Coeff x^1
        %     sigma0  = std(Ndata)/2;
        %     beta0   = p(4); % Intercetta
        % 
        %     % I coefficienti di un polinomio POSSONO essere negativi.
        %     % Impostare lb a -Inf è fondamentale.
        %     lb = [-Inf, -Inf, -Inf, 1e-5, -Inf];
        %     ub = [ Inf,  Inf,  Inf, std(Ndata)*100, Inf];
        % case 'powerlaw'
        %     % N(t) = N0 + alpha * t^gamma
        %     % Mapping: N0 -> N0, K -> alpha, lambda -> gamma
        %     N00 = Ndata(1);       % N0 (Intercetta)
        %     K0 = 1.0;             % Alpha (Scale)
        %     lambda0 = 1.0;        % Gamma (Exponent)
        %     sigma0 = std(Ndata)/2;
        %     beta0 = 0;            % Unused
        % 
        %     lb = [0, -Inf, 0, 1e-3, -Inf]; % Alpha può essere negativo se decresce
        %     ub = [10, Inf, max(Ndata)*2, std(Ndata)*10, Inf];