function plot_prob_choice_by_rdiff(reward_diff_summary_table,study_info)

    % === SETUP ===
    fig = figure('Name','Interactive Prob Plot','Position',[100, 100, 1200, 700]);

    rdiffs = reward_diff_summary_table.reward_diff;
    choice_nums = 1:study_info.num_choices_big_hor;
    horizons = [1, study_info.num_free_choices_big_hor];
    all_colors = lines(numel(choice_nums) * numel(horizons));

    ax1 = subplot(2,1,1); hold on;
    title('P(Choose Right) by Reward Difference');
    xlabel('Reward Difference'); ylabel('P(Choose Right)');

    ax2 = subplot(2,1,2); hold on;
    title('P(Choose High-Info) by Reward Difference');
    xlabel('Reward Difference'); ylabel('P(High-Info)');

    % Store line handles + labels
    line_pairs = {};
    labels = {};
    checkbox_idx = 0;
    line_idx = 0;

    % === PLOTTING LINES ===
    for h = horizons
        for c = choice_nums
            line_idx = line_idx + 1;

            %% Top plot: P(Choose Right)
            col_top = sprintf('mean_prob_choose_right_hor%d_choice%d', h, c);
            top_line = gobjects(1);
            if ismember(col_top, reward_diff_summary_table.Properties.VariableNames)
                y_top = reward_diff_summary_table.(col_top);
                std_col_top = sprintf('std_prob_choose_right_hor%d_choice%d', h, c);
                if ismember(std_col_top, reward_diff_summary_table.Properties.VariableNames)
                    err_top = reward_diff_summary_table.(std_col_top);
                    top_line = errorbar(ax1, rdiffs, y_top, err_top, ...
                        '-', 'Color', all_colors(line_idx,:), 'LineWidth', 1.5, 'CapSize', 3);
                else
                    top_line = plot(ax1, rdiffs, y_top, '-', ...
                        'Color', all_colors(line_idx,:), 'LineWidth', 1.5);
                end
            end

            %% Bottom plot: P(Choose High-Info)
            col_bot = sprintf('mean_prob_high_info_hor%d_choice%d', h, c);
            bot_line = gobjects(1);
            if ismember(col_bot, reward_diff_summary_table.Properties.VariableNames)
                y_bot = reward_diff_summary_table.(col_bot);
                std_col_bot = sprintf('std_prob_high_info_hor%d_choice%d', h, c);
                if ismember(std_col_bot, reward_diff_summary_table.Properties.VariableNames)
                    err_bot = reward_diff_summary_table.(std_col_bot);
                    bot_line = errorbar(ax2, rdiffs, y_bot, err_bot, ...
                        '-', 'Color', all_colors(line_idx,:), 'LineWidth', 1.5, 'CapSize', 3);
                else
                    bot_line = plot(ax2, rdiffs, y_bot, '-', ...
                        'Color', all_colors(line_idx,:), 'LineWidth', 1.5);
                end
            end

            if isgraphics(top_line) || isgraphics(bot_line)
                checkbox_idx = checkbox_idx + 1;
                line_pairs{checkbox_idx} = {top_line, bot_line};
                labels{checkbox_idx} = sprintf('Hor=%d, Choice=%d', h, c);
            end
        end
    end

    % === INLINE LEGEND ===
    legend_ax = axes('Parent', fig, ...
                     'Position', [0.78, 0.1, 0.2, 0.3]); % adjust as needed
    axis(legend_ax, 'off'); hold(legend_ax, 'on');

    for i = 1:numel(line_pairs)
        plot(legend_ax, [0, 1], [1-i, 1-i], '-', ...
             'Color', line_pairs{i}{1}.Color, 'LineWidth', 2);
        text(legend_ax, 1.1, 1-i, labels{i}, ...
             'Interpreter', 'none', 'FontSize', 9, 'VerticalAlignment', 'middle');
    end
    xlim(legend_ax, [0, 3]); ylim(legend_ax, [-numel(line_pairs), 1]);

    % === CHECKBOXES ===
    figure(fig); % ensure UI stays on main figure
    for i = 1:numel(line_pairs)
        uicontrol('Parent', fig, ...
                  'Style','checkbox', ...
                  'String', labels{i}, ...
                  'Value', 1, ...
                  'Units','normalized', ...
                  'Position',[0.01, 0.95 - (i*0.03), 0.15, 0.025], ...
                  'FontSize', 9, ...
                  'Callback', @(src,~) toggle_lines(src, line_pairs{i}));
    end
end

function toggle_lines(src, line_pair)
    for j = 1:2
        if isgraphics(line_pair{j})
            set(line_pair{j}, 'Visible', logical2onoff(src.Value));
        end
    end
end

function str = logical2onoff(val)
    str = 'off';
    if val, str = 'on'; end
end
