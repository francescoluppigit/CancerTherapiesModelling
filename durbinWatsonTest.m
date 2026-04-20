function [dw_stat] = durbinWatsonTest(residuals)
% Calculates the Durbin-Watson statistic for serial correlation.
    if isempty(residuals) || length(residuals) < 2
        dw_stat = NaN;
        return;
    end
    e = residuals(:); % Ensure column vector
    diff_e = diff(e);
    dw_stat_numerator = sum(diff_e.^2);
    dw_stat_denominator = sum(e.^2);
    
    if dw_stat_denominator == 0
        dw_stat = NaN; % Avoid division by zero
    else
        dw_stat = dw_stat_numerator / dw_stat_denominator;
    end
end