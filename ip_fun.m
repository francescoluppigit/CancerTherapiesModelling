function[ip] = ip_fun(mle, md)
    lam = mle(1); K = mle(2); N0 = mle(3);
            if K > N0 && lam > 0
                if contains(md, 'logistic'), ip = (1/lam) * log((K - N0)/N0);
                elseif contains(md, 'gompertz'), ip = (1/lam) * log(log(K/N0));
                elseif contains(md, 'richards') && mle(5) > 0
                    v = ((K/N0)^mle(5) - 1) / mle(5);
                    if v > 0, ip = (1/(mle(5)*lam)) * log(v); end
                end
            end
end