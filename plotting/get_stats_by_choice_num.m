
function summary_table = get_stats_by_choice_num(MDP, simmed_model_output)
    % Get the gen mean difference for each game
    left_means = mean(MDP.bandit1_schedule(:,1:4), 2); % get mean of forced choices on left
    right_means = mean(MDP.bandit2_schedule(:,1:4), 2); % get mean of forced choices on right
    gen_mean_diff = round(right_means - left_means);
    actions = simmed_model_output.actions;
    [n_games, n_trials] = size(actions);
    % Replicate gen_mean_diff to match shape
    gen_mean_diff_matrix = repmat(gen_mean_diff, 1, n_trials);
    % Find where the person chose the high mean option
    chose_high_mean_option = zeros(size(actions));
    % Apply logic
    chose_high_mean_option(actions == 1 & gen_mean_diff_matrix < 0) = 1;
    chose_high_mean_option(actions == 2 & gen_mean_diff_matrix > 0) = 1;
    % Set NaNs where action is NaN
    chose_high_mean_option(isnan(actions)) = nan;
    % Compute probability of choosing high mean option
    prob_chose_high_mean_option = simmed_model_output.action_probs;
    prob_chose_high_mean_option(chose_high_mean_option == 0) = 1 - prob_chose_high_mean_option(chose_high_mean_option == 0);

    % Compute probability of choosing high info option
    high_info_option = nan(size(actions));
    for i = 1:n_games
        for t = 2:n_trials
            past = actions(i,1:t-1);
            if all(isnan(past)), continue; end
            count1 = sum(past == 1, 'omitnan');
            count2 = sum(past == 2, 'omitnan');
            if count1 < count2
                high_info_option(i,t) = 1;
            elseif count2 < count1
                high_info_option(i,t) = 2;
            end
        end
    end
    % First get the probability of choosing the right option
    prob_choose_right = simmed_model_output.action_probs; % initialize with action probs
    prob_choose_right(simmed_model_output.actions == 1) = 1 - prob_choose_right(simmed_model_output.actions == 1);
    % Compute the probability of choosing the high info option
    prob_choose_high_info = prob_choose_right; % initialize this to probability of choosing right, then invert when left was high info
    prob_choose_high_info(high_info_option == 1) = 1 - prob_choose_high_info(high_info_option == 1);
    prob_choose_high_info(isnan(high_info_option)) = nan;


    % Compute horizon values: row-wise count of non-NaNs in MDP.actions minus 4
    horizon = sum(~isnan(MDP.actions), 2) - 4;
    unique_horizons = unique(horizon);

    % Initialize struct
    summary = struct();
    summary.choice_num = (1:n_trials)';
    n_choices = n_trials;
    for h = 1:numel(unique_horizons)
        h_val = unique_horizons(h);
        % Identify rows of the current horizon
        rows = horizon == h_val;
        % Iterate through choices for each game of the specified
        % horizon
        for c = 1:n_choices
            % Probability of choosing high mean
            prob_high_mean = prob_chose_high_mean_option(rows, c);
            summary.(['mean_prob_choose_cor_hor' num2str(h_val)])(c,1) = mean(prob_high_mean, 'omitnan');
            summary.(['std_prob_choose_cor_hor' num2str(h_val)])(c,1)  = std(prob_high_mean,'omitnan');
                    
            % Probability of choosing high info option
            prob_high_info_vals = prob_choose_high_info(rows, c);
            summary.(['mean_prob_high_info_hor' num2str(h_val)])(c,1) = mean(prob_high_info_vals,'omitnan');
            summary.(['std_prob_high_info_hor' num2str(h_val)])(c,1)  = std(prob_high_info_vals,'omitnan');

            % Reaction time
            rt_vals = simmed_model_output.rts(rows, c);
            summary.(['mean_rt_hor' num2str(h_val)])(c,1) = mean(rt_vals,'omitnan');
            summary.(['std_rt_hor' num2str(h_val)])(c,1)  = std(rt_vals,'omitnan');      
        end
    end
    % Convert to table (optional)
    summary_table = struct2table(summary);
    summary_table = summary_table(:, ~all(ismissing(summary_table), 1));
end