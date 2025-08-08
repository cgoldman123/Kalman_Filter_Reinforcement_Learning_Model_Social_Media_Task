function ff = social_model_free(processed_data, simulated_data)
    num_games = processed_data.num_games;
    num_forced_choices = processed_data.num_forced_choices;
    num_free_choices_big_hor = processed_data.num_free_choices_big_hor;
    num_choices_big_hor = num_forced_choices + num_free_choices_big_hor;

    % Get generative mean diffs
    if isfield(processed_data, 'bandit1_mean') && isfield(processed_data,'bandit2_mean')
        left_means = processed_data.bandit1_mean; % get gen mean
        right_means = processed_data.bandit2_mean; % get gen mean
    else
        left_means = mean(processed_data.bandit1_schedule(:,1:4), 2); % get mean of forced choices on left
        right_means = mean(processed_data.bandit2_schedule(:,1:4), 2); % get mean of forced choices on right
    end

    gen_mean_diff = round(right_means - left_means);
    % Get unique values and preallocate
    [unique_rdiffs, ~, idx_rdiff] = unique(gen_mean_diff);
    gen_mean_diffs = unique_rdiffs'; % generative mean differences

    % Build data structure to do model free analyses
    data(num_games,1) = struct(); % creates an empty struct array 

    % If passing in simulated data, replace participants' actual behavior
    % with simulated behavior
    if isempty(fieldnames(simulated_data))
        actions = processed_data.actions;
        rewards = processed_data.rewards;
        RTs = processed_data.RTs;

    else
        actions = simulated_data.actions;
        rewards = simulated_data.rewards;
        RTs = simulated_data.RTs;
    end

    for game = 1:num_games
        data(game).game_num = game - 1;
        data(game).key = actions(game, ~isnan(actions(game, :)));
        data(game).num_forced_left = sum(data(game).key(1:4) == 1);
        data(game).gameLength = length(data(game).key);
        data(game).nfree = data(game).gameLength - processed_data.num_forced_choices;
        data(game).horizon = data(game).nfree;
        data(game).reward = rewards(game, ~isnan(rewards(game, :)));
        data(game).mean = [left_means(game); right_means(game)];
        data(game).mean_diff = round(left_means(game) - right_means(game));
        data(game).max_side = (right_means(game) > left_means(game))+1; % Should be 2 when right side is better, 1 when left
        % Save the schedule of both bandits
        data(game).rewards = [processed_data.bandit1_schedule(game, ~isnan(rewards(game, :))); processed_data.bandit2_schedule(game, ~isnan(rewards(game, :)))];
        data(game).correct = data(game).key == ((data(game).rewards(1,:) < data(game).rewards(2,:)) + 1); % Checks if subject is choosing the better option according to schedule
        data(game).correcttot = sum(data(game).correct(5:end));
        data(game).accuracy = data(game).correcttot/data(game).nfree;

        data(game).mean_correct = data(game).key == ((data(game).mean(1) < data(game).mean(2)) + 1);

        data(game).left_observed = mean(data(game).reward(data(game).key==1));
        data(game).right_observed = mean(data(game).reward(data(game).key==2));

        left_observed_mean_before_choice5 = mean(data(game).reward(data(game).key(1:4)==1));
        right_observed_mean_before_choice5 = mean(data(game).reward(data(game).key(1:4)==2));

        data(game).choice5_generative_correct = data(game).key(5) == data(game).max_side;
        data(game).choice5_true_correct = data(game).choice5_generative_correct;
        data(game).choice5_observed_correct = data(game).key(5) == ((left_observed_mean_before_choice5 <right_observed_mean_before_choice5) + 1);
        
       left_observed_mean_before_last_choice = mean(data(game).reward(data(game).key(1:end-1)==1));
       right_observed_mean_before_last_choice = mean(data(game).reward(data(game).key(1:end-1)==2));

        data(game).last_generative_correct = data(game).key(end) == data(game).max_side;
        data(game).last_true_correct = data(game).last_generative_correct;
        data(game).last_observed_correct = data(game).key(end) == ((left_observed_mean_before_last_choice < right_observed_mean_before_last_choice) + 1);
        
        
        data(game).got_total = sum(data(game).reward(data(game).key==2)) + sum(data(game).reward(data(game).key==1));
        for rt_index=1:length(RTs(game,:))
            data(game).RT(rt_index) = RTs(game,rt_index);
        end
        data(game).RT_choice5 = data(game).RT(5);
        data(game).RT_choiceLast = data(game).RT(data(game).gameLength);
        data(game).true_correct_frac = sum(data(game).mean_correct(5:end))/length(data(game).mean_correct(5:end));
    end


    %%%%%%%%%%%%%%%% Statistics related to accuracy %%%%%%%%%%%%%%%%

    big_hor = data([data.horizon] == num_free_choices_big_hor);
    small_hor = data([data.horizon] == 1);
    
    small_hor_13 = small_hor([small_hor.num_forced_left]~=2);

    big_hor_13 = big_hor([big_hor.num_forced_left]~=2);
    
    big_hor_meancor = vertcat(big_hor.mean_correct);
    small_hor_meancor = vertcat(small_hor.mean_correct);
    
    % Store free choice accuracy in big horizon games in each of free
    % choices
    for i = 1:num_free_choices_big_hor
        col_idx = i + 4; % Column indices that correspond to free choices 
        fieldname = sprintf('big_hor_freec%d_acc', i); % Dynamically create the field name
        ff.(fieldname) = sum(big_hor_meancor(:, col_idx)) / numel(big_hor); % Compute accuracy for each free choice
    end

    ff.small_hor_freec1_acc = sum(small_hor_meancor(:, 5)) / numel(small_hor);
   

    %%%%%%%%%%%%%%%% Probability of choosing left side given reward difference %%%%%%%%%%%%%%%%

    % First load in mean diffs as a cell string
    mean_diffs = arrayfun(@(x) sprintf('%02d', abs(x)), gen_mean_diffs, 'UniformOutput', false);
    mean_diffs(gen_mean_diffs < 0) = cellfun(@(s) ['-' s], mean_diffs(gen_mean_diffs < 0), 'UniformOutput', false);
    for mean_diff = mean_diffs
        mean_diff_double = str2double(mean_diff);
        mean_diff_char = mean_diff{:};
        if mean_diff_double > 0
            more_or_less = 'more';
        else
            more_or_less = 'less';
        end

        % Do this for small horizon  
        % first filter to games of the mean difference
        filtered_dat = small_hor_13([small_hor_13.mean_diff] == mean_diff_double);  
        n = numel(filtered_dat); % Get number of games
        sum_val = 0; % Initialize counter for times chose left
        % Loop through each element and check if the last key is 1
        for i = 1:n
            sum_val = sum_val + (filtered_dat(i).key(end) == 1); % chose left
        end
        % Compute average probability of choosing left given gen mean
        % difference
        ff.(['small_hor_13_left_' mean_diff_char(end-1:end) '_' more_or_less '_prob']) = sum_val / n;
        
        % Do this for big horizon
        % Filter data based on matching mean_diff
        filtered_dat = big_hor_13([big_hor_13.mean_diff] == mean_diff_double);  
        % Loop through the free choices 
        for free_choice_num = 1:num_free_choices_big_hor
            col_idx = free_choice_num + 4; % maps free_choice_num = 1 to key(5), etc.
            fieldname = sprintf('big_hor_13_left_%s_%s_choice_%d_prob',mean_diff_char(end-1:end), more_or_less, free_choice_num);        
            n = numel(filtered_dat); % Get number of games
            sum_val = 0; % Initialize counter for times chose left
            % Loop through each element and check if the free choice key is 1
            for i = 1:n
                sum_val = sum_val + (filtered_dat(i).key(col_idx) == 1); % chose left
            end
            ff.(fieldname) = sum_val / n;
        end
    end


    %%%%%%%%%%%%%%%% Probability of choosing the high info side  %%%%%%%%%%%%%%%%

    % Get the probability of choosing the high info side when
    % it's generative mean is more/less than the low info side
    % For all free choices
    result_struct = struct();
    for game_num = 1:length(data)
        game = data(game_num);
        choices = game.key;
        if game.horizon == 1
            choice_indices = 5;  % only one free choice in small horizon
        else
            choice_indices = 5:num_choices_big_hor;  % 5 to 9 for big horizon
        end
    
        for choice_num = choice_indices
            % determine if high/low info choice
            num_1_choices = sum(choices(1:choice_num-1) == 1);
            num_2_choices = sum(choices(1:choice_num-1) == 2);
            if choices(choice_num) == 1
                made_high_info_choice = num_1_choices < num_2_choices;
                made_low_info_choice  = num_1_choices > num_2_choices;
            else
                made_high_info_choice = num_1_choices > num_2_choices;
                made_low_info_choice  = num_1_choices < num_2_choices;
            end
            if made_high_info_choice || made_low_info_choice
                % generative mean difference between high and low info option
                if num_1_choices > num_2_choices
                    gen_mean_diff = round(game.mean(2) - game.mean(1));
                else
                    gen_mean_diff = round(game.mean(1) - game.mean(2));
                end
                % Format gen_mean_diff
                if gen_mean_diff < 0
                    gen_mean_char = sprintf('%02d_less', abs(gen_mean_diff));
                else
                    gen_mean_char = sprintf('%02d_more', gen_mean_diff);
                end
                % Determine if this is H1 or H5
                prefix = 'big_hor_';
                if game.horizon == 1
                    prefix = 'small_hor_';
                end
                % Create count and total field names
                count_field = sprintf('%smore_info_%s_choice_%d_count', prefix, gen_mean_char, choice_num - 4);
                total_field = sprintf('%smore_info_%s_choice_%d_total', prefix, gen_mean_char, choice_num - 4);
                % Initialize if necessary
                if ~isfield(result_struct, count_field)
                    result_struct.(count_field) = 0;
                    result_struct.(total_field) = 0;
                end
                % Update counts
                result_struct.(count_field) = result_struct.(count_field) + made_high_info_choice;
                result_struct.(total_field) = result_struct.(total_field) + 1;
            end
        end
    end


    % Initialize counters for collapsed info across mean differences
    for i = 1:num_free_choices_big_hor
        collapsed.count(i) = 0;
        collapsed.total(i) = 0;
    end
    fields = fieldnames(result_struct);
    for i = 1:numel(fields)
        field_name = fields{i};        
        % Compute per-condition probability and store in ff
        if contains(field_name, 'count')
            base_name = strrep(field_name, '_count', '');
            ff.([base_name '_prob']) = result_struct.(field_name) / result_struct.([base_name '_total']);
        end
        % Collapse across generative mean differences for each choice
        for c = 1:num_free_choices_big_hor
            if contains(field_name, sprintf('choice_%d_count', c))
                collapsed.count(c) = collapsed.count(c) + result_struct.(field_name);
            elseif contains(field_name, sprintf('choice_%d_total', c))
                collapsed.total(c) = collapsed.total(c) + result_struct.(field_name);
            end
        end
    end

    % Average across all small_hor high info probs
    probs = [];
    for i = 1:length(gen_mean_diffs)
        amt = gen_mean_diffs(i);
        if amt < 0
            suffix = sprintf('%02d_less', abs(amt));
        else
            suffix = sprintf('%02d_more', amt);
        end
        fieldname = ['small_hor_more_info_' suffix '_choice_1_prob'];
        if isfield(ff, fieldname)
            probs(end+1) = ff.(fieldname);
        end
    end
    ff.small_hor_more_info_prob = mean(probs);

    % Add NaN placeholders for missing fields in ff
    gen_mean_diffs_str = unique(arrayfun(@(x) sprintf('%02d', abs(x)), gen_mean_diffs, 'UniformOutput', false));
    choice_nums = arrayfun(@num2str, 1:num_free_choices_big_hor, 'UniformOutput', false);
    for mean_diff = gen_mean_diffs_str
        for choice_num = choice_nums
            % Check and fill "more" condition
            field = ['big_hor_more_info_' mean_diff{:} '_more_choice_' choice_num{:} '_prob'];
            if ~isfield(ff, field)
                ff.(field) = NaN;
            end
            % Check and fill "less" condition
            field = ['big_hor_more_info_' mean_diff{:} '_less_choice_' choice_num{:} '_prob'];
            if ~isfield(ff, field)
                ff.(field) = NaN;
            end
        end
    end

    % Compute probability of choosing high info side for each choice number
    for i = 1:num_free_choices_big_hor
        fieldname = sprintf('big_hor_more_info_choice_%d_prob', i);
        ff.(fieldname) = collapsed.count(i) / collapsed.total(i);
    end

    %%%%%%%%%%%%%%%% Run a t test to determine if someone was value sensitive %%%%%%%%%%%%%%%%

    % get generative mean difference (left - right) for left and right
    % choices, respectively, to determine if value sensitive
    gen_mean_diff_for_right_choices = []; % should be negative
    gen_mean_diff_for_left_choices = []; % should be positive
    for game_num = 1:length(data)
        game = data(game_num,:);
        choices = game.key;
        for (choice_num = 5:length(choices))
            choice = choices(choice_num);
            if choice == 1
                gen_mean_diff_for_left_choices = [gen_mean_diff_for_left_choices game.mean(1) - game.mean(2)];
            else
                gen_mean_diff_for_right_choices = [gen_mean_diff_for_right_choices game.mean(1) - game.mean(2)];
            end
        end
    end
    % t test
    % significance means that person was value-sensitive in the appropriate
    % direction i.e., left-right for left choices was greater than
    % left-right for right choices
    [h, p, ci, stats] = ttest2(gen_mean_diff_for_left_choices, gen_mean_diff_for_right_choices,'Tail', 'right');
    ff.p_value_of_t_test_for_value_sensitivity = p;
    
    
    % ---------------------------------------------------------------
    
    ff.mean_RT       = mean([data.RT],'omitnan');
    ff.sub_accuracy  = mean([data.accuracy]);
    
    ff.choice5_acc_gen_mean      = mean([data.choice5_generative_correct]);
    ff.choice5_acc_obs_mean      = mean([data.choice5_observed_correct]);
    ff.choice5_acc_true_mean     = mean([data.choice5_true_correct]);
    ff.choice5_acc_gen_mean_big_hor   = mean([big_hor.choice5_generative_correct]);
    ff.choice5_acc_obs_mean_big_hor   = mean([big_hor.choice5_observed_correct]);
    ff.choice5_acc_true_mean_big_hor  = mean([big_hor.choice5_true_correct]);
    ff.choice5_acc_gen_mean_small_hor   = mean([small_hor.choice5_generative_correct]);
    ff.choice5_acc_obs_mean_small_hor   = mean([small_hor.choice5_observed_correct]);
    ff.choice5_acc_true_mean_small_hor  = mean([small_hor.choice5_true_correct]);
    
    ff.last_acc_gen_mean         = mean([data.last_generative_correct]);
    ff.last_acc_obs_mean         = mean([data.last_observed_correct]);
    ff.last_acc_true_mean        = mean([data.last_true_correct]);
    ff.last_acc_gen_mean_big_hor      = mean([big_hor.last_generative_correct]);
    ff.last_acc_obs_mean_big_hor      = mean([big_hor.last_observed_correct]);
    ff.last_acc_true_mean_big_hor     = mean([big_hor.last_true_correct]);
    ff.last_acc_gen_mean_small_hor      = mean([small_hor.last_generative_correct]);
    ff.last_acc_obs_mean_small_hor      = mean([small_hor.last_observed_correct]);
    ff.last_acc_true_mean_small_hor     = mean([small_hor.last_true_correct]);

    
    ff.mean_RT_big_hor                = mean([big_hor.RT],'omitnan'); 
    ff.mean_RT_small_hor                = mean([small_hor.RT],'omitnan'); 
    
    ff.mean_RT_choice5           = mean([data.RT_choice5]);
    ff.mean_RT_choiceLast        = mean([data.RT_choiceLast]);
    
    ff.mean_RT_choice5_big_hor        = mean([big_hor.RT_choice5]);
    ff.mean_RT_choiceLast_big_hor     = mean([big_hor.RT_choiceLast]);
    ff.mean_RT_choice5_small_hor        = mean([small_hor.RT_choice5]);
    ff.mean_RT_choiceLast_small_hor     = mean([small_hor.RT_choiceLast]);
    
    ff.true_correct_frac         = mean([data.true_correct_frac]);
    ff.true_correct_frac_small_hor      = mean([small_hor.true_correct_frac]);
    ff.true_correct_frac_big_hor      = mean([big_hor.true_correct_frac]);

    ff.num_games                 = size(data,1);
    
end
