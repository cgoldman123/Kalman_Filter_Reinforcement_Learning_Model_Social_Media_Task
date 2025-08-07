function plot_total_uncert_and_estimated_rdiff(reward_diff_summary_table,study_info)
    figure;

    % Define constants
    choice_nums = 1:study_info.num_choices_big_hor;
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
        for h = [1, study_info.num_free_choices_big_hor]
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
                ls = '-'; % solid for small horizon
            else
                ls = '--'; % dotted for big horizon
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
        for h = [1, study_info.num_free_choices_big_hor]
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
end