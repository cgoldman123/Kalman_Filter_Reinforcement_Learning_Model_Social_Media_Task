
function make_plots_model_statistics(reward_diff_summary_table, choice_num_summary_table)
    figure;

    % Define constants
    choice_nums = 1:9;
    pos_rdiff_indices = height(reward_diff_summary_table)/2+1 : height(reward_diff_summary_table); % only get indices of positive rdiffs when using absolute value of rdiff
    unique_rdiffs = reward_diff_summary_table.reward_diff(pos_rdiff_indices);
    num_rdiffs = numel(unique_rdiffs);
    colors = lines(num_rdiffs); % one color per reward diff
    legends = {};

    % --- Subplot 1: Total Uncertainty ---
    subplot(2,1,1); hold on;
    title('Total Uncertainty by Choice Number');
    xlabel('Choice Number');
    ylabel('Total Uncertainty');

    for ri = 1:num_rdiffs
        i = pos_rdiff_indices(ri);
        rdiff = reward_diff_summary_table.reward_diff(i);
        for h = [1, 5]
            y_vals = zeros(1, numel(choice_nums));
            for c = choice_nums
                col_name = sprintf('mean_total_uncert_hor%d_choice%d', h, c);
                if ismember(col_name, reward_diff_summary_table.Properties.VariableNames)
                    y_vals(c) = reward_diff_summary_table.(col_name)(i);
                else
                    y_vals(c) = nan;
                end
            end
            if h == 1
                ls = '-'; % solid for horizon 1
            else
                ls = '--'; % dotted for horizon 5
            end
            plot(choice_nums, y_vals, ls, 'Color', colors(ri,:), 'LineWidth', 1.5);
            legends{end+1} = sprintf('RD=%d, Hor=%d', rdiff, h);
        end
    end
    legend(legends, 'Location', 'bestoutside');
    grid on;

    % --- Subplot 2: Estimated Reward Difference ---
    subplot(2,1,2); hold on;
    title('Estimated Reward Difference by Choice Number');
    xlabel('Choice Number');
    ylabel('Estimated Reward Difference');

    for ri = 1:num_rdiffs
        i = pos_rdiff_indices(ri);
        rdiff = reward_diff_summary_table.reward_diff(i);
        for h = [1, 5]
            y_vals = zeros(1, numel(choice_nums));
            for c = choice_nums
                col_name = sprintf('mean_est_mean_diff_hor%d_choice%d', h, c);
                if ismember(col_name, reward_diff_summary_table.Properties.VariableNames)
                    y_vals(c) = reward_diff_summary_table.(col_name)(i);
                else
                    y_vals(c) = nan;
                end
            end
            if h == 1
                ls = '-';
            else
                ls = '--';
            end
            plot(choice_nums, y_vals, ls, 'Color', colors(ri,:), 'LineWidth', 1.5);
        end
    end
    legend(legends, 'Location', 'bestoutside');
    grid on;




    figure;

    rdiffs = reward_diff_summary_table.reward_diff;
    choice_nums = 1:9;
    horizons = [1, 5];

    % Generate one color per horizon-choice combo
    num_colors = numel(choice_nums) * numel(horizons);
    all_colors = lines(num_colors);

    % --- Subplot 1: Mean RT ---
    subplot(2,1,1); hold on;
    title('Mean rt by reward difference');
    xlabel('Reward Difference');
    ylabel('Mean RT');

    legends = {};
    line_idx = 0;
    for h = horizons
        for c = choice_nums
            line_idx = line_idx + 1;
            col_name = sprintf('mean_rt_hor%d_choice%d', h, c);
            if ismember(col_name, reward_diff_summary_table.Properties.VariableNames)
                y_vals = reward_diff_summary_table.(col_name);
                plot(rdiffs, y_vals, '-', 'Color', all_colors(line_idx,:), 'LineWidth', 1.5);
                legends{end+1} = sprintf('Hor=%d, Choice=%d', h, c);
            end
        end
    end
    legend(legends, 'Location', 'bestoutside');
    grid on;

    % --- Subplot 2: Mean RT for high - low info ---
    subplot(2,1,2); hold on;
    title('Mean rt by reward difference for high - low info option');
    xlabel('Reward Difference');
    ylabel('Mean RT (high - low info)');

    legends = {};
    line_idx = 0;
    for h = horizons
        for c = choice_nums
            line_idx = line_idx + 1;
            col_name = sprintf('mean_rt_high_minus_low_info_hor%d_choice%d', h, c);
            if ismember(col_name, reward_diff_summary_table.Properties.VariableNames)
                y_vals = reward_diff_summary_table.(col_name);
                plot(rdiffs, y_vals, '-', 'Color', all_colors(line_idx,:), 'LineWidth', 1.5);
                legends{end+1} = sprintf('Hor=%d, Choice=%d', h, c);
            end
        end
    end
    legend(legends, 'Location', 'bestoutside');
    grid on;

    figure;

    choice_nums = choice_num_summary_table.choice_num;
    horizons = [1, 5];
    colors = lines(numel(horizons)); % one color per horizon

    %% --- Subplot 1: Probability of Correct Choice ---
    subplot(3,1,1); hold on;
    title('Probability of Correct Choice');
    xlabel('Choice Number');
    ylabel('P(Correct)');
    legends = {};
    for h = 1:numel(horizons)
        h_val = horizons(h);
        mean_col = sprintf('mean_prob_choose_cor_hor%d', h_val);
        std_col  = sprintf('std_prob_choose_cor_hor%d', h_val);
        if ismember(mean_col, choice_num_summary_table.Properties.VariableNames)
            y_vals   = choice_num_summary_table.(mean_col);
            y_err    = choice_num_summary_table.(std_col);
            errorbar(choice_nums, y_vals, y_err, '-o', ...
                'Color', colors(h,:), 'LineWidth', 1.5, 'MarkerSize', 4);
            legends{end+1} = sprintf('Horizon %d', h_val);
        end
    end
    legend(legends, 'Location', 'best');
    grid on;

    %% --- Subplot 2: Reaction Time ---
    subplot(3,1,2); hold on;
    title('Reaction Time');
    xlabel('Choice Number');
    ylabel('Mean RT');
    legends = {};
    for h = 1:numel(horizons)
        h_val = horizons(h);
        mean_col = sprintf('mean_rt_hor%d', h_val);
        std_col  = sprintf('std_rt_hor%d', h_val);
        if ismember(mean_col, choice_num_summary_table.Properties.VariableNames)
            y_vals   = choice_num_summary_table.(mean_col);
            y_err    = choice_num_summary_table.(std_col);
            errorbar(choice_nums, y_vals, y_err, '-o', ...
                'Color', colors(h,:), 'LineWidth', 1.5, 'MarkerSize', 4);
            legends{end+1} = sprintf('Horizon %d', h_val);
        end
    end
    legend(legends, 'Location', 'best');
    grid on;

    %% --- Subplot 3: Probability of Choosing High-Info ---
    subplot(3,1,3); hold on;
    title('Probability of Choosing High-Info Option');
    xlabel('Choice Number');
    ylabel('P(High Info)');
    legends = {};
    for h = 1:numel(horizons)
        h_val = horizons(h);
        mean_col = sprintf('mean_prob_high_info_hor%d', h_val);
        std_col  = sprintf('std_prob_high_info_hor%d', h_val);
        if ismember(mean_col, choice_num_summary_table.Properties.VariableNames)
            y_vals   = choice_num_summary_table.(mean_col);
            y_err    = choice_num_summary_table.(std_col);
            errorbar(choice_nums, y_vals, y_err, '-o', ...
                'Color', colors(h,:), 'LineWidth', 1.5, 'MarkerSize', 4);
            legends{end+1} = sprintf('Horizon %d', h_val);
        end
    end
    legend(legends, 'Location', 'best');
    grid on;



end