function plot_behavior_by_choice_num(choice_num_summary_table,study_info)

    figure;

    choice_nums = choice_num_summary_table.choice_num;
    horizons = [1, study_info.num_free_choices_big_hor];
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