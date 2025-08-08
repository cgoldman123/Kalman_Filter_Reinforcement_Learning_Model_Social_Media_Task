function processed_data = process_behavioral_data_exercise(raw_data)

    % ---- constants (match your .mat layout) ----
    NUM_FORCED   = 4;
    NUM_FREE_BIG = 6;                 % total stored cols = 10
    NCOLS        = NUM_FORCED + NUM_FREE_BIG;

    vn = raw_data.Properties.VariableNames;
    need = {'game_number','game_type','force_pos','left_reward','right_reward'};
    miss = setdiff(need, vn);
    if ~isempty(miss)
        error('process_behavioral_data_SM:MissingCols', ...
              'Missing columns: %s', strjoin(miss, ', '));
    end
    if ~ismember('resp_keys', vn), raw_data.resp_keys = repmat({''}, height(raw_data), 1); end
    if ~ismember('resp_rt', vn),   raw_data.resp_rt   = nan(height(raw_data),1);          end
    if ~ismember('dislike_room', vn), raw_data.dislike_room = zeros(height(raw_data),1);  end

    % coerce stuff that might be strings
    toNum = @(x) (isnumeric(x) * x) + (~isnumeric(x)) * str2double(string(x));
    raw_data.left_reward  = toNum(raw_data.left_reward);
    raw_data.right_reward = toNum(raw_data.right_reward);

    if ismember('trial_num', vn)
        raw_data = sortrows(raw_data, {'game_number','trial_num'});
    else
        raw_data = sortrows(raw_data, {'game_number','trials_thisN'});
    end
    if ~iscell(raw_data.force_pos), raw_data.force_pos = cellstr(raw_data.force_pos); end
    if ~iscell(raw_data.resp_keys), raw_data.resp_keys = cellstr(raw_data.resp_keys); end

    games = unique(raw_data.game_number);
    num_games = numel(games);

    % ---- prealloc as doubles ----
    actions           = NaN(num_games, NCOLS);
    RTs               = NaN(num_games, NCOLS);
    rewards           = NaN(num_games, NCOLS);
    bandit1_schedule  = NaN(num_games, NCOLS);   % left: actual rewards
    bandit2_schedule  = NaN(num_games, NCOLS);   % right: actual rewards
    horizon_type      = NaN(num_games, 1);
    forced_info_diff  = NaN(num_games, 1);

    for gi = 1:num_games
        G = raw_data(raw_data.game_number == games(gi), :);

        % 1 = small (h1), 2 = big (h6) — anything not 'h1' counted big
        horizon_type(gi) = 1 + double(~strcmpi(G.game_type{1}, 'h1'));

        % info diff from first 4 forced trials:  -(#R - 2)
        fp = upper(string(G.force_pos));
        nR = sum(fp(1:min(NUM_FORCED, numel(fp))) == "R");
        if nR == 3 || nR == 1
            forced_info_diff(gi) = -1;
        else
            forced_info_diff(gi) = 1;
        end

        % dislike inversion flag
        is_dislike = toNum(G.dislike_room(1)) == 1;

        n = min(height(G), NCOLS);
        for t = 1:n
            % actual per-trial rewards, with dislike inversion if needed
            Lr = G.left_reward(t);
            Rr = G.right_reward(t);
            if is_dislike
                Lr = 100 - Lr; Rr = 100 - Rr;
            end

            % action: forced uses force_pos; free uses resp_keys
            fpos = upper(string(G.force_pos(t)));
            if fpos == "L"
                a = 1; rt = NaN;
            elseif fpos == "R"
                a = 2; rt = NaN;
            else
                rk = lower(string(G.resp_keys(t)));
                if rk == "left"
                    a = 1;
                elseif rk == "right"
                    a = 2;
                else
                    a = NaN;
                end
                rt = G.resp_rt(t);
            end

            actions(gi, t) = a;
            RTs(gi, t)     = rt;

            % schedules are the ACTUAL rewards shown on each side
            bandit1_schedule(gi, t) = Lr;   % left
            bandit2_schedule(gi, t) = Rr;   % right

            % realized reward follows chosen side
            if a == 1
                rewards(gi, t) = Lr;
            elseif a == 2
                rewards(gi, t) = Rr;
            else
                rewards(gi, t) = NaN;
            end
        end
    end

    processed_data = struct( ...
        'horizon_type',             horizon_type, ...
        'num_games',                num_games, ...
        'num_forced_choices',       NUM_FORCED, ...
        'num_free_choices_big_hor', NUM_FREE_BIG, ...
        'forced_choice_info_diff',  forced_info_diff, ...
        'actions',                  actions, ...
        'RTs',                      RTs, ...
        'rewards',                  rewards, ...
        'bandit1_schedule',         bandit1_schedule, ...
        'bandit2_schedule',         bandit2_schedule );
end
