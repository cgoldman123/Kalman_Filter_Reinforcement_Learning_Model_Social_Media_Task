function data = parse_table(table, subject, run, num_runs, room_type)
    if nargin < 5
        NUM_GAMES = 80; %for social media
    else
        NUM_GAMES = num_runs;
    end
       
    if size(table,1) ==0
       error('Data table is of size 0')
       
    end

    trial_nums = unique(table.trial_number);
    trialvar = 'trial_number';
    if trial_nums < 10
        if ~isnan(str2double(table.bandit_num))
            table.bandit_num = str2double(table.bandit_num);
        end
        trial_nums = unique(table.bandit_num);
        trialvar = 'bandit_num';
    end
    for game_number=0:NUM_GAMES-1
        trial = find(trial_nums==game_number);
        try
            gamedata = table(table.(trialvar) == trial_nums(trial), :);
        catch
            continue
        end
        gameLength = size(gamedata, 1);
        if ~ismember(gameLength, [5 9])
            continue
        end
        forced = cellfun(@(x) x ~= 'X', gamedata.force_pos)';
        free   = ~forced;
        nforced = numel(forced(forced > 0));
        nfree = gameLength - nforced;
        
        means = [gamedata.left_mean(1); gamedata.right_mean(1)];
                       
        rewards = [gamedata.left_reward gamedata.right_reward]';
        if strcmp(room_type, "Dislike")
            rewards = 100-rewards;
            means = 100-means;
        end      
        gID = -1; % I have no idea what gID is -- gameID?
        key = cellfun(@(x) strcmp(x, 'right') + 1, gamedata.response)';
        reward = diag(rewards(key, :))';

        RT = gamedata.response_time';      
        RT_choice5 = RT(5);
        RT_choiceLast = RT(end);

        [~, lr_correct] = max(rewards, [], 1);
        [~, max_side] = max(means, [], 1);
        max_points = sum(max(rewards));

        correct = key == lr_correct;
        correcttot = sum(correct(end-nfree+1:end));
        accuracy = correcttot / nfree;
        
        mean_correct = key == max_side;
        
        forced_choices = key(forced);
        forced_left  = sum(forced_choices == 1);
        forced_right = sum(forced_choices == 2);
        forced_type = [forced_left forced_right];
        
        [~, more_info] = min(forced_type);
        [~, less_info] = max(forced_type);
        
        if forced_left == forced_right
            more_info = 0;
            info_diff = means(2) - means(1);
%             info_diff = mean(rewards(2, (key == 2) & forced)) - mean(rewards(1, (key == 1) & forced));
        else
            info_diff = means(more_info) - means(less_info);
%             info_diff = mean(rewards(more_info, forced)) - mean(rewards(less_info, forced));
        end
        
        got_points = sum(reward);
        
        left_observed = sum(rewards(1, key(1:4) == 1)) / sum(key(1:4) == 1);
        right_observed = sum(rewards(2, key(1:4) == 2)) / sum(key(1:4) == 2);
        
        left_true  = sum(rewards(1, 1:4)) / 4;
        right_true = sum(rewards(2, 1:4)) / 4;

        [~, max_observed_side] = max([left_observed, right_observed]);
        [~, max_true_side]     = max([left_true, right_true]);
        
        choice5_generative_correct  = key(5) == max_side;
        choice5_observed_correct    = key(5) == max_observed_side;
        choice5_true_correct        = key(5) == max_true_side;
                
        left_observed = sum(rewards(1, key(1:(end-1)) == 1)) / sum(key(1:(end-1)) == 1);
        right_observed = sum(rewards(2, key(1:(end-1)) == 2)) / sum(key(1:(end-1)) == 2);
        
        left_true  = sum(rewards(1, :)) / gameLength; %should it be 1:end-1
        right_true = sum(rewards(2, :)) / gameLength;

        [~, max_observed_side] = max([left_observed, right_observed]);
        [~, max_true_side]     = max([left_true, right_true]);
        
        last_generative_correct     = key(end) == max_side;
        last_observed_correct       = key(end) == max_observed_side;
        last_true_correct           = key(end) == max_true_side;
        
        true_correct_count = sum(key(5:end) == max_true_side) / (gameLength - 4);
        subs = regexp(subject, '[A-Z]{2}[0-9]{3}', 'match');
        if isempty(subs)
            subs = extractBetween(subject, 'social_media_', '_T');
        end

        data(game_number + 1) = struct(         ...
            'subject', subs{1},                 ...
            'game_num', game_number,            ...
            'run', run,                         ...
            'gameLength', gameLength,           ...
            'nforced', nforced,                 ...
            'forced', forced,                   ...
            'nfree', nfree,                     ...
            'free', free,                       ...
            'mean', means,                      ...
            'rewards', rewards,                 ...
            'key', key,                         ...
            'reward', reward,                   ...
            'RT', RT,                           ...
            'correct', correct,                 ...
            'correcttot', correcttot,           ...
            'accuracy', accuracy,               ...
            'horizon', nfree,                   ...
            'mean_correct', mean_correct,        ...
            'max_side', max_side,               ...
            'forced_type', forced_type,         ...
            'more_info', more_info,             ...
            'info_diff', info_diff,             ...
                                                ...
            'left_observed', left_observed,                             ...
            'right_observed', right_observed,                           ...
            'max_total', max_points,                                    ...
            'got_total', got_points,                                    ...
            'choice5_generative_correct', choice5_generative_correct,   ...
            'choice5_observed_correct', choice5_observed_correct,       ...
            'choice5_true_correct', choice5_true_correct,               ...
            'last_generative_correct', last_generative_correct,         ...
            'last_observed_correct', last_observed_correct,             ...
            'last_true_correct', last_true_correct,                     ...
            'RT_choice5', RT_choice5,                                   ...
            'RT_choiceLast', RT_choiceLast,                             ...
            'true_correct_frac', true_correct_count                     ...
        );
    end  
    good_index = []; 
    for k=1:size(data,2)
        good_index = [good_index ~isempty(struct2table(data).subject{k})];
    end

    temp_table = struct2table(data);
    data = temp_table(good_index==1,:);

    if size(data,1) < 20 
        error('Not enough games')
    end
    data = table2struct(data);

end