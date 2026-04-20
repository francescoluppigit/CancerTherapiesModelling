function [nLL] = likelihood_wrapper(model_type, params, t, Ndata)
    % Helper function to call the correct likelihood
    % This is used by both fit_growth_model and the profiling loop

    lambda = params(1); K = params(2); N0 = params(3); 
    sigma = params(4); beta = params(5);

    switch lower(model_type)
        case 'logistic'
            LL = likelihood_Logistic(lambda, K, N0, sigma, t, Ndata);
        case 'logisticlogn'
            LL = likelihood_LogisticLogN(lambda, K, N0, sigma, t, Ndata);
        case 'gompertz'
            LL = likelihood_Gompertz(lambda, K, N0, sigma, t, Ndata);
        case 'gompertzlogn'
            LL = likelihood_GompertzLogN(lambda, K, N0, sigma, t, Ndata);
        case 'richards'
            LL = likelihood_Richards(lambda, K, N0, sigma, beta, t, Ndata);
        case 'richardslogn'
            LL = likelihood_RichardsLogN(lambda, K, N0, sigma, beta, t, Ndata);
        case 'exponential'
            LL = likelihood_Exponential(lambda, N0, sigma, beta, t, Ndata);        
        otherwise
            error('Unknown model type: %s', model_type);
    end
    nLL = -LL; % Return Negative Log-Likelihood
end

%% OTHER MODELS
% case 'polynomial'
%             % n(1)=a3, n(2)=a2, n(3)=a1, n(4)=sigma, n(5)=a0
%             LL = likelihood_Polynomial(lambda, K, N0, sigma, beta, t, Ndata);
%         case 'vonbertalanffy'
%             % n(1)=gamma, n(2)=K, n(3)=N0, n(4)=sigma
%             LL = likelihood_VonBertalanffy(lambda, K, N0, sigma, t, Ndata);
%         case 'biexponential'
%             % n(1)=alpha, n(2)=A, n(3)=beta, n(4)=sigma, n(5)=B
%             LL = likelihood_BiExponential(lambda, K, N0, sigma, beta, t, Ndata);        
%         case 'powerlaw'
%             % n(1)=gamma, n(2)=alpha, n(3)=N0, n(4)=sigma
%             LL = likelihood_PowerLaw(lambda, K, N0, sigma, t, Ndata);