function [raw_data, subject_data_info] = get_raw_data_exercise(root, id, run_num,room)
% Read one participant/run CSV, return cleaned trial table and a subject_data_info struct.

    % ---------- normalize run tag ----------
    if isnumeric(run_num)
        run_tag = sprintf('T%d', run_num);
    else
        run_tag = char(run_num);
        if ~startsWith(run_tag,'T'), run_tag = ['T' run_tag]; end
    end

    % ---------- find all matching files for this run ----------
    patt = fullfile(root, 'NPC','DataSink','StimTool_Online','Exercise_Study', ...
                    sprintf('horizon_%s_%s_*.csv', id, run_tag));
    hits = dir(patt);
    if isempty(hits)
        error('get_raw_data:NoFile','No CSV found for %s %s under %s', id, run_tag, root);
    end
    % newest is used as the main file we load
    [~,ixNewest] = max([hits.datenum]);
    csv_file = fullfile(hits(ixNewest).folder, hits(ixNewest).name);

    % ---------- load the chosen file (robust to garbage lines/quotes) ----------
    T = read_horizon_csv(csv_file);

    % required columns
    need = {'game_number','trials_thisN'};
    for k = 1:numel(need)
        if ~ismember(need{k}, T.Properties.VariableNames)
            error('get_raw_data:BadCSV','Missing expected column "%s" in %s', need{k}, csv_file);
        end
    end

    % normalize key columns to numeric
    game_num = to_num(T.game_number);
    trialN   = to_num(T.trials_thisN);

    % keep only real trial rows
    mask_trials = ~isnan(game_num) & ~isnan(trialN);
    raw_data = T(mask_trials, :);

    % sort by game & trial order
    if ismember('trial_num', raw_data.Properties.VariableNames)
        raw_data = sortrows(raw_data, {'game_number','trial_num'});
    else
        raw_data = sortrows(raw_data, {'game_number','trials_thisN'});
    end

    % ---------- filter to like or dislike rooms ----------
    is_dislike = strcmp(room,'Dislike') || strcmp(room,'dislike');
    raw_data = raw_data(raw_data.dislike_room==is_dislike,:);


    % practice effects: count how many files for this run contain trials
    n_with_trials = 0;
    for i = 1:numel(hits)
        f = fullfile(hits(i).folder, hits(i).name);
        try
            Ti = read_horizon_csv(f);
            if all(ismember(need, Ti.Properties.VariableNames))
                gn = to_num(Ti.game_number);
                tn = to_num(Ti.trials_thisN);
                if any(~isnan(gn) & ~isnan(tn))
                    n_with_trials = n_with_trials + 1;
                end
            end
        catch
            % ignore unreadable/garbage files for practice-effects count
        end
    end
    has_practice_effects = double(n_with_trials > 1);

    subject_data_info = struct( ...
        'id',                  id, ...
        'has_practice_effects',has_practice_effects, ...
        'cb',                  run_tag, ...
        'behavioral_file_path',csv_file, ...
        'room_type',           room );

end

% ================= helper functions =================

function T = read_horizon_csv(csv_file)
    % Find the header line that actually contains the expected tokens.
    hdr = detect_header_line(csv_file);
    if isempty(hdr)
        error('get_raw_data:HeaderNotFound', ...
              'Could not find header with "game_number" and "trials.thisN" in %s', csv_file);
    end
    % Read using that header; old-MATLAB compatible.
    opts = detectImportOptions(csv_file,'FileType','text','Delimiter',',');
    opts.VariableNamesLine = hdr;
    opts.DataLines         = [hdr+1, Inf];
    opts.ExtraColumnsRule  = 'ignore';
    opts.EmptyLineRule     = 'read';
    T = readtable(csv_file, opts);

    % strip leading/trailing quotes from any text columns
    for v = 1:width(T)
        if iscellstr(T.(v))
            T.(v) = regexprep(T.(v), '^"|"$', '');
        elseif isstring(T.(v))
            T.(v) = regexprep(cellstr(T.(v)), '^"|"$', '');
        end
    end

    % make MATLAB-safe names (turn "trials.thisN" into "trials_thisN", etc.)
    T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);

    % be tolerant to slight naming variants
    T = rename_if_match(T, {'^game[_\.]*number$','^gamenumber$'}, 'game_number');
    T = rename_if_match(T, {'^trials[_\.]*thisN$','^trialsThisN$','^trial.*thisN$'}, 'trials_thisN');
    T = rename_if_match(T, {'^trial[_\.]*num$','^trialNum$'}, 'trial_num');
end

function idx = detect_header_line(file)
    fid = fopen(file,'r'); if fid < 0, idx = []; return; end
    i = 0; idx = [];
    while true
        line = fgetl(fid); if ~ischar(line), break; end
        i = i + 1; L = lower(line);
        if contains(L,'game_number') && (contains(L,'trials.thisn') || contains(L,'trials_thisn'))
            idx = i; break
        end
    end
    fclose(fid);
end

function T = rename_if_match(T, patterns, target)
    v = T.Properties.VariableNames;
    for p = 1:numel(patterns)
        m = find(~cellfun('isempty', regexpi(v, patterns{p})), 1, 'first');
        if ~isempty(m), T.Properties.VariableNames{m} = target; return; end
    end
end

function x = to_num(col)
    if isnumeric(col), x = double(col);
    elseif iscellstr(col), x = str2double(col);
    else, x = str2double(string(col));
    end
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
