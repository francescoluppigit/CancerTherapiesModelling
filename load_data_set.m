function [data, curve, L] = load_data_set(data_folder)
    % Automatically detects the dominant data file extension in the folder
    % and loads all files of that type.
    % OUTPUT: data (cell array of file paths), curve (cell array of file names), 
    %         L (count), final_extension (detected extension)
    
    % Define relevant data extensions to consider
    valid_exts = {'.csv', '.dat', '.txt', '.xlsx', '.xls'};
    
    % Search for ALL files in the directory
    all_files = dir(data_folder);
    
    % Filter files to count occurrences of valid extensions
    ext_counts = containers.Map('KeyType', 'char', 'ValueType', 'int32');
    
    for k = 1:length(all_files)
        file_name = all_files(k).name;
        if ~all_files(k).isdir && file_name(1) ~= '.' % Skip directories and hidden files
            [~, ~, ext] = fileparts(file_name);
            ext = lower(ext); % Standardize extension
            
            if ismember(ext, valid_exts)
                if isKey(ext_counts, ext)
                    ext_counts(ext) = ext_counts(ext) + 1;
                else
                    ext_counts(ext) = 1;
                end
            end
        end
    end

if isempty(ext_counts)
    error('No valid data files found in folder: %s. Valid types are: %s', ...
          data_folder, strjoin(valid_exts, ', '));
end

% Determine the main extension
ext_keys = keys(ext_counts);
ext_values = cell2mat(values(ext_counts));
[max_count, max_idx] = max(ext_values);
final_extension = ext_keys{max_idx};

fprintf('Automatically detected file extension: %s (Found %d files).\n', final_extension, max_count);

% Load the files using the detected extension
search_pattern = fullfile(data_folder, ['*' final_extension]);
file_list = dir(search_pattern);

L = length(file_list);
data = cell(L, 1);
curve = cell(L, 1);

for k = 1:L
    data{k} = fullfile(file_list(k).folder, file_list(k).name);
    [~, name, ~] = fileparts(file_list(k).name);
    curve{k} = name;
end

disp(curve);
end