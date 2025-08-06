function ff = social_model_free(root,file, room_type, study,simulated_data)
    num_games_in_roomtype = 40;
    num_total_games = num_games_in_roomtype*2;
    num_forced_choices = 4;
    num_free_choices_big_hor = 5;
    num_choices_big_hor = num_forced_choices + num_free_choices_big_hor;
    gen_mean_diffs = [-24, -12, -8, -4, -2, 2, 4, 8, 12, 24]; % generative mean differences


    %The only difference between the two versions of the schedules is the
    %order of blocks. Since we fit Dislike and Like rooms separately and
    %the same block within each room type always goes first (within a
    %session), we can use the same schedule.
    
     % determine if cb=1 or cb=2
    if strcmp(study,'local')
        if contains(file, '_R1-')
            schedule = readtable([root 'rsmith/wellbeing/tasks/SocialMedia/schedules/sm_distributed_schedule_CB1.csv']);
            cb = 1;
        else
            schedule = readtable([root 'rsmith/wellbeing/tasks/SocialMedia/schedules/sm_distributed_schedule_CB2.csv']);
            cb = 2;
        end
    elseif strcmp(study,'prolific')
        if contains(file, '_CB_')
            schedule = readtable([root 'rsmith/wellbeing/tasks/SocialMedia/schedules/sm_distributed_schedule_CB2.csv']);
            cb = 2;
        else
           schedule = readtable([root 'rsmith/wellbeing/tasks/SocialMedia/schedules/sm_distributed_schedule_CB1.csv']);
            cb = 1;
        end
     end

    orgfunc = str2func(['Social_' study '_organize']);
    subj_data = orgfunc(file, schedule, room_type);

    % for debugging: disp(subj_data);disp(file{:});disp(ses); disp(room_type);
    ses = 999; % filler because session is no longer relevant
    data = parse_table(subj_data, file, ses, num_total_games, room_type);
        
    % Calculate the difference between left and right options
    mean_diff_values = arrayfun(@(x) x.mean(1) - x.mean(2), data);
    for i = 1:numel(data)
        data(i).mean_diff = mean_diff_values(i);
    end


    % If passing in simulated data, replace participants' actual behavior
    % with simulated behavior in "data"
    if ~isempty(fieldnames(simulated_data))
        for game = 1:num_games_in_roomtype
            data(game).key = simulated_data.actions(game, ~isnan(simulated_data.actions(game, :)));
            data(game).reward = simulated_data.rewards(game, ~isnan(simulated_data.rewards(game, :)));
            data(game).accuracy = data(game).correcttot/data(game).nfree;
            data(game).correct = data(game).key == ((data(game).rewards(1,:) < data(game).rewards(2,:)) + 1);
            data(game).correcttot = sum(data(game).correct(5:end));
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
            for rt_index=1:length(data(game).RT)
                data(game).RT(rt_index) = simulated_data.RTs(game,rt_index);
            end
            data(game).RT_choice5 = data(game).RT(5);
            data(game).RT_choiceLast = data(game).RT(end);
            data(game).true_correct_frac = sum(data(game).mean_correct(5:end))/length(data(game).mean_correct(5:end));
        end
    end

    % ff = fit_horizon(data, ses, room_type);
    ff.room_type = room_type;
    ff.counterbalance = cb;


    %%%%%%%%%%%%%%%% Statistics related to accuracy %%%%%%%%%%%%%%%%

    big_hor = data([data.horizon] == num_free_choices_big_hor);
    small_hor = data([data.horizon] == 1);
    
    %small_hor_22 = small_hor(sum([vertcat(small_hor.forced_type)'] == 2) == 2);
    small_hor_13 = small_hor(sum([vertcat(small_hor.forced_type)'] == 2) ~= 2);

    %h6_22 = h6(sum([vertcat(h6.forced_type)'] == 2) == 2);
    big_hor_13 = big_hor(sum([vertcat(big_hor.forced_type)'] == 2) ~= 2);
    
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
   

    %%%%%%%%%%%%%%%% Probability of choosing a side given reward difference %%%%%%%%%%%%%%%%

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
        filtered_dat = small_hor_13([small_hor_13.mean_diff] == mean_diff_double);  
        ff.(['small_hor_left_' mean_diff_char(end-1:end) '_' more_or_less '_prob']) = ((filtered_dat(1).key(end) == 1) + (filtered_dat(2).key(end) == 1))/2;
        % Do this for big horizon
        % Filter data based on matching mean_diff
        filtered_dat = big_hor_13([big_hor_13.mean_diff] == mean_diff_double);  
        % Loop through the free choices 
        for i = 1:num_free_choices_big_hor
            col_idx = i + 4; % maps i = 1 to key(5), etc.
            fieldname = sprintf('big_hor_left_%s_%s_choice_%d_prob',mean_diff_char(end-1:end), more_or_less, i);
            ff.(fieldname) = ((filtered_dat(1).key(col_idx) == 1) + (filtered_dat(2).key(col_idx) == 1)) / 2;
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
                    gen_mean_diff = game.mean(2) - game.mean(1);
                else
                    gen_mean_diff = game.mean(1) - game.mean(2);
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
    
    ff.mean_RT       = mean([data.RT]);
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

    
    ff.mean_RT_big_hor                = mean([big_hor.RT]); 
    ff.mean_RT_small_hor                = mean([small_hor.RT]); 
    
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
