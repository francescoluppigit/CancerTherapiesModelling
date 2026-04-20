function [t, Ndata] = import_data(file)
    % Data import function: selects method based on file extension.
    [~, ~, file_ext] = fileparts(file);
    switch lower(file_ext)
        case {'.xlsx', '.xls'}
            % --- CASE 1: Excel Files ---
            try
                % Forcing data lines to 1 ensures we read all data if no header is present
                opts = detectImportOptions(file);
                opts.DataLines = [1, Inf];
                M = readmatrix(file, opts);
            catch ME
                error('Error reading Excel file %s: %s', file, ME.message);
            end
            
        case {'.csv', '.dat', '.txt'}
            % --- CASE 2: Text Files (European Format assumed) ---
            % Delimiter = Semicolon (;), Decimal Separator = Comma (,)
            try
                opts = detectImportOptions(file, 'Delimiter', ';');
                opts = setvaropts(opts, 'Type', 'double');
                opts = setvaropts(opts, 'DecimalSeparator', ',');
                opts.DataLines = [1, Inf]; 
                
                M = readmatrix(file, opts);
            catch ME
                error('Error reading file %s (check Delimiter/Decimal Separator): %s', file, ME.message);
            end
            
        otherwise
            error('Unsupported file extension: %s. Update import_data function.', file_ext);
    end
    % --- Data Extraction and Validation ---
    if isempty(M) || size(M, 2) < 2
        error('No data read or file %s has fewer than 2 columns.', file);
    end
    t = M(:, 1); Ndata = M(:, 2);
end
