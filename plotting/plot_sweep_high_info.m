function plot_sweep_high_info(simmed_model_output,study_info)

    % === Identify swept parameter ===
    n_params = size(simmed_model_output, 1);
    param_structs = simmed_model_output(:,1);
    all_fields = fieldnames(param_structs{1});

    sweep_param = '';
    sweep_vals = zeros(n_params,1);
    for f = 1:numel(all_fields)
        vals = cellfun(@(s) s.(all_fields{f}), param_structs);
        if numel(unique(vals)) > 1
            sweep_param = all_fields{f};
            sweep_vals = vals(:);
            break;
        end
    end
    if isempty(sweep_param)
        error('Could not determine swept parameter.');
    end

    % === SETUP ===
    fig = figure('Name','Parameter Sweep High Info','Position',[100, 100, 1200, 700]);

    choice_nums = 1:study_info.num_choices_big_hor;
    horizons = [1, study_info.num_free_choices_big_hor];
    all_colors = lines(numel(choice_nums) * numel(horizons));

    % Prepare axes
    ax = axes(fig); hold on;
    xlabel(ax, sweep_param);
    ylabel(ax, 'P(Choose High Info)');
    title(ax, ['P(Choose High Info) across values of ', sweep_param]);
    grid(ax, 'on');

    % Store valid lines and labels
    line_pairs = {};  % {main_line, dummy_line}
    labels = {};
    checkbox_idx = 0;
    line_idx = 0;

    % === PLOTTING LINES ===
    for h = horizons
        for c = choice_nums
            line_idx = line_idx + 1;
            col_mean = sprintf('mean_prob_high_info_hor%d', h);
            col_std  = sprintf('std_prob_high_info_hor%d', h);

            % Skip if column doesn't exist in the table
            if ~ismember(col_mean, simmed_model_output{1,3}.Properties.VariableNames)
                continue
            end

            means = NaN(n_params,1);
            stds  = NaN(n_params,1);
            for i = 1:n_params
                stats_tbl = simmed_model_output{i,3};
                row = stats_tbl.choice_num == c;
                if any(row)
                    means(i) = stats_tbl{row, col_mean};
                    stds(i)  = stats_tbl{row, col_std};
                end
            end

            % Skip if all are NaN
            if all(isnan(means))
                continue
            end

            % Plot line
            h_line = errorbar(ax, sweep_vals, means, stds, ...
                '-', 'Color', all_colors(line_idx,:), 'LineWidth', 1.5, 'CapSize', 3);
            checkbox_idx = checkbox_idx + 1;
            line_pairs{checkbox_idx} = {h_line, gobjects(1)};
            labels{checkbox_idx} = sprintf('Hor=%d, Choice=%d', h, c);
        end
    end

    % === INLINE LEGEND ===
    legend_ax = axes('Parent', fig, ...
                     'Position', [0.78, 0.1, 0.2, 0.3]);
    axis(legend_ax, 'off'); hold(legend_ax, 'on');

    for i = 1:numel(line_pairs)
        plot(legend_ax, [0, 1], [1-i, 1-i], '-', ...
            'Color', line_pairs{i}{1}.Color, 'LineWidth', 2);
        text(legend_ax, 1.1, 1-i, labels{i}, ...
            'Interpreter', 'none', 'FontSize', 9, 'VerticalAlignment', 'middle');
    end
    xlim(legend_ax, [0, 3]); ylim(legend_ax, [-numel(line_pairs), 1]);

    % === CHECKBOXES ===
    figure(fig); % ensure we’re back on the main plot for UI
    for i = 1:numel(line_pairs)
        uicontrol('Parent', fig, ...
                  'Style','checkbox', ...
                  'String', labels{i}, ...
                  'Value', 1, ...
                  'Units','normalized', ...
                  'Position',[0.01, 0.95 - (i*0.03), 0.18, 0.025], ...
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
