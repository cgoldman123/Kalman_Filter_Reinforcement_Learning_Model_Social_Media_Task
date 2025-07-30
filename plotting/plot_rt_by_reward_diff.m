function plot_rt_by_reward_diff(reward_diff_summary_table)

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
end