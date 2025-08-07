
function summary_table = get_stats_by_reward_diff(MDP, simmed_model_output)
    left_means = mean(MDP.bandit1_schedule(:,1:4), 2); % get mean of forced choices on left
    right_means = mean(MDP.bandit2_schedule(:,1:4), 2); % get mean of forced choices on right
    gen_mean_diff = round(right_means - left_means);
    % Get unique values and preallocate
    [unique_rdiffs, ~, idx_rdiff] = unique(gen_mean_diff);
    n_rdiffs = numel(unique_rdiffs);
    n_choices = size(simmed_model_output.total_uncertainty, 2);
    
    % Compute horizon values: row-wise count of non-NaNs in MDP.actions minus 4
    horizon = sum(~isnan(MDP.actions), 2) - 4;
    unique_horizons = unique(horizon);

    % Compute probability of choosing right option
    prob_choose_right = simmed_model_output.action_probs; % initialize with action probs
    prob_choose_right(simmed_model_output.actions == 1) = 1 - prob_choose_right(simmed_model_output.actions == 1);

    % Compute probability of choosing high info option
    actions = simmed_model_output.actions;
    [n_games, n_trials] = size(actions);
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
    prob_choose_high_info = prob_choose_right; % initialize this to probability of choosing right, then invert when left was high info
    prob_choose_high_info(high_info_option == 1) = 1 - prob_choose_high_info(high_info_option == 1);
    prob_choose_high_info(isnan(high_info_option)) = nan;

    % Compute generative reward difference for high info - low info
    % Replicate gen_mean_diff to match the shape of high_info_option
    gen_mean_diff_matrix = repmat(gen_mean_diff, 1, size(high_info_option, 2));
    % Copy and flip where high_info_option is 1 (since gen_mean_diff is
    % right - left, no change is necessary when right is high_info_option)
    gen_mean_diff_high_info_minus_low_info = gen_mean_diff_matrix;
    gen_mean_diff_high_info_minus_low_info(high_info_option == 1) = -gen_mean_diff_high_info_minus_low_info(high_info_option == 1);

    
    % Initialize struct
    summary = struct();
    summary.reward_diff = unique_rdiffs;
    
    % Loop through reward diffs and horizons
    for i = 1:n_rdiffs
        % Identify rows of the current reward difference
        rows_rdiff = find(idx_rdiff == i);
        % Identify rows of the current absolute value reward difference
        rows_abs_rdiff = find(idx_rdiff == i | idx_rdiff == (11 - i));
        rdiff = unique_rdiffs(i);
    
        for h = 1:numel(unique_horizons)
            h_val = unique_horizons(h);
            % Identify rows of the current horizon
            rows_rdiff_hor = rows_rdiff(horizon(rows_rdiff) == h_val);
            rows_abs_rdiff_hor = rows_abs_rdiff(horizon(rows_abs_rdiff) == h_val);

            for c = 1:n_choices
                % For total uncertainty and estimated mean reward
                % difference, only consider the absolute value of reward
                % difference (i.e., i > 5 and rdiff > 0)
                if (rdiff > 0)
                    % Uncertainty
                    uncert_vals = simmed_model_output.total_uncertainty(rows_abs_rdiff_hor, c);
                    summary.(['mean_total_uncert_hor' num2str(h_val) '_choice' num2str(c)])(i,1) = mean(uncert_vals,'omitnan');
                    summary.(['std_total_uncert_hor' num2str(h_val) '_choice' num2str(c)])(i,1)  = std(uncert_vals,'omitnan');
        
                    % Estimated mean difference
                    est_vals = simmed_model_output.estimated_mean_diff(rows_abs_rdiff_hor, c);
                    abs_est_vals = abs(est_vals);
                    summary.(['mean_est_mean_diff_hor' num2str(h_val) '_choice' num2str(c)])(i,1) = mean(abs_est_vals,'omitnan');
                    summary.(['std_est_mean_diff_hor' num2str(h_val) '_choice' num2str(c)])(i,1)  = std(abs_est_vals,'omitnan');
                else
                    % Put NaN values in negative reward differences
                    summary.(['mean_total_uncert_hor' num2str(h_val) '_choice' num2str(c)])(i,1) = nan;
                    summary.(['std_total_uncert_hor' num2str(h_val) '_choice' num2str(c)])(i,1)  = nan;
                    summary.(['mean_est_mean_diff_hor' num2str(h_val) '_choice' num2str(c)])(i,1) = nan;
                    summary.(['std_est_mean_diff_hor' num2str(h_val) '_choice' num2str(c)])(i,1)  = nan;
                end

                % Probability of choosing right
                prob_right_vals = prob_choose_right(rows_rdiff_hor, c);
                summary.(['mean_prob_choose_right_hor' num2str(h_val) '_choice' num2str(c)])(i,1) = mean(prob_right_vals,'omitnan');
                summary.(['std_prob_choose_right_hor' num2str(h_val) '_choice' num2str(c)])(i,1)  = std(prob_right_vals,'omitnan');
                        
                % Probability of choosing high info option for high - low
                % info rdiff
                prob_high_info_vals_for_choice_num = prob_choose_high_info(horizon == h_val, c);
                prob_high_info_vals_for_choice_num_and_rdiff = prob_high_info_vals_for_choice_num(gen_mean_diff_high_info_minus_low_info(horizon == h_val,c)==rdiff);
                summary.(['mean_prob_high_info_hor' num2str(h_val) '_choice' num2str(c)])(i,1) = mean(prob_high_info_vals_for_choice_num_and_rdiff,'omitnan');
                summary.(['std_prob_high_info_hor' num2str(h_val) '_choice' num2str(c)])(i,1)  = std(prob_high_info_vals_for_choice_num_and_rdiff,'omitnan');

                % Reaction time
                rt_vals = simmed_model_output.rts(rows_rdiff_hor, c);
                summary.(['mean_rt_hor' num2str(h_val) '_choice' num2str(c)])(i,1) = mean(rt_vals,'omitnan');
                summary.(['std_rt_hor' num2str(h_val) '_choice' num2str(c)])(i,1)  = std(rt_vals,'omitnan');          

                % Reaction time for generative mean difference of high info
                % minus low info rdiff
                rt_vals_for_choice_num = simmed_model_output.rts(horizon == h_val, c);
                rt_vals_for_choice_num_and_rdiff = rt_vals_for_choice_num(gen_mean_diff_high_info_minus_low_info(horizon == h_val,c)==rdiff);
                summary.(['mean_rt_high_minus_low_info_hor' num2str(h_val) '_choice' num2str(c)])(i,1) = mean(rt_vals_for_choice_num_and_rdiff,'omitnan');
                summary.(['std_rt_high_minus_low_info_hor' num2str(h_val) '_choice' num2str(c)])(i,1)  = std(rt_vals_for_choice_num_and_rdiff,'omitnan');  
            end
        end
    end
    
    % Convert to table (optional)
    summary_table = struct2table(summary);
    summary_table = summary_table(:, ~all(ismissing(summary_table), 1));

end