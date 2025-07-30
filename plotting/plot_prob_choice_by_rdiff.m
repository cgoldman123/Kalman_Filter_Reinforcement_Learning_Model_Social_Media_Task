function plot_prob_choice_by_rdiff(reward_diff_summary_table)
    figure;

    % Setup
    rdiffs = reward_diff_summary_table.reward_diff;
    choice_nums = 1:9;
    horizons = [1, 5];
    num_lines = numel(choice_nums) * numel(horizons);
    colors = lines(num_lines); % one color per horizon-choice combo

    %% --- Subplot 1: P(Choose Right) ---
    subplot(2,1,1); hold on;
    title('P(Choose Right) by Reward Difference');
    xlabel('Reward Difference');
    ylabel('P(Choose Right)');
    legends = {};
    line_idx = 0;

    for h = horizons
        for c = choice_nums
            line_idx = line_idx + 1;
            col_name = sprintf('mean_prob_choose_right_hor%d_choice%d', h, c);
            if ismember(col_name, reward_diff_summary_table.Properties.VariableNames)
                y_vals = reward_diff_summary_table.(col_name);
                plot(rdiffs, y_vals, '-', 'Color', colors(line_idx,:), 'LineWidth', 1.5);
                legends{end+1} = sprintf('Hor %d, Choice %d', h, c);
            end
        end
    end
    legend(legends, 'Location', 'bestoutside');
    grid on;

    %% --- Subplot 2: P(Choose High-Info) ---
    subplot(2,1,2); hold on;
    title('P(Choose High-Info) by Reward Difference');
    xlabel('Reward Difference');
    ylabel('P(High-Info)');
    legends = {};
    line_idx = 0;

    for h = horizons
        for c = choice_nums
            line_idx = line_idx + 1;
            col_name = sprintf('mean_prob_high_info_hor%d_choice%d', h, c);
            if ismember(col_name, reward_diff_summary_table.Properties.VariableNames)
                y_vals = reward_diff_summary_table.(col_name);
                plot(rdiffs, y_vals, '-', 'Color', colors(line_idx,:), 'LineWidth', 1.5);
                legends{end+1} = sprintf('Hor %d, Choice %d', h, c);
            end
        end
    end
    legend(legends, 'Location', 'bestoutside');
    grid on;
end
