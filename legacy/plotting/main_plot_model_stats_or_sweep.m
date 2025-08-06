function main_plot_model_stats_or_sweep(root, fitting_procedure, experiment, room_type, results_dir,MDP, id)
    % First call get_fits to get the schedule/forced choices before
    MDP.get_processed_behavior_and_dont_fit_model = 1; % Toggle on to extract the rts and other processed behavioral data but not fit the model
    MDP.fit_model = 1; % Toggle on even though the model won't fit
    [rt_data, mdp] = get_fits(root, fitting_procedure, experiment,room_type, results_dir, MDP, id);
    
    % Load the mdp variable to get bandit schedule
    % load(['./SPM_scripts/social_media_' experiment '_mdp_cb' num2str(cb) '.mat']); 

    mdp_fieldnames = fieldnames(mdp);
    for (i=1:length(mdp_fieldnames))
        MDP.(mdp_fieldnames{i}) = mdp.(mdp_fieldnames{i});
    end
    actions_and_rts.actions = mdp.actions;
    actions_and_rts.RTs = nan(40,9);

    if isempty(MDP.param_to_sweep) 
        param_values_to_sweep_over = 1;
    else
        param_values_to_sweep_over = MDP.param_values_to_sweep_over;
    end

    % Sweep through parameter values if doing a sweep
    for (param_set_idx=1:length(param_values_to_sweep_over))
        if ~isempty(MDP.param_to_sweep) 
            MDP.params.(MDP.param_to_sweep) = MDP.param_values_to_sweep_over(param_set_idx);
        end
        % Simulate behavior using max pdf or take a bunch of samples
        if MDP.num_samples_to_draw_from_pdf == 0
            simmed_model_output{param_set_idx,1} = MDP.params; % save the parameters used to simulate behavior
            model_results = MDP.model(MDP.params, actions_and_rts, MDP.rewards, MDP, 1); % save the behavior
            reward_diff_summary_table = get_stats_by_reward_diff(MDP, model_results);
            choice_num_summary_table = get_stats_by_choice_num(MDP, model_results);
            simmed_model_output{param_set_idx,2} = reward_diff_summary_table;
            simmed_model_output{param_set_idx,3} = choice_num_summary_table;
        else
            for sample_num = 1:MDP.num_samples_to_draw_from_pdf
                simmed_model_output{param_set_idx,sample_num,1} = MDP.params; % save the parameters used to simulate behavior
                model_results = MDP.model(MDP.params, actions_and_rts, MDP.rewards, MDP, 1); % save the behavior
                samp_reward_diff_summary_table{sample_num} = get_stats_by_reward_diff(MDP, model_results);
                samp_choice_num_summary_table{sample_num} = get_stats_by_choice_num(MDP, model_results);
            end
            % Build choice_num_summary_table by averaging sampled values
            % Initialize with non-mean/std columns from the first table
            base_tbl = samp_choice_num_summary_table{1};
            vars = base_tbl.Properties.VariableNames;
            keep = ~(startsWith(vars, 'mean_') | startsWith(vars, 'std_'));
            choice_num_summary_table = base_tbl(:, keep);
            % Loop through mean_ columns only
            mean_vars = vars(startsWith(vars, 'mean_'));
            for v = mean_vars
                col = v{1};
                % Stack column across all 3 tables
                vals_mat = cell2mat(cellfun(@(t) t.(col), samp_choice_num_summary_table, 'UniformOutput', false));
                
                % Compute mean and std across columns (samples)
                choice_num_summary_table.(col) = mean(vals_mat, 2, 'omitnan');
                % Calculate std of sampled means
                % Create std_ column name by replacing mean_ with std_
                std_col = ['std_' extractAfter(col, 'mean_')];
                choice_num_summary_table.(std_col) = std(vals_mat, 0, 2, 'omitnan');
            end

            % Build reward_diff_summary_table by averaging sampled values
            % Initialize with non-mean/std columns from the first table
            base_tbl = samp_reward_diff_summary_table{1};
            vars = base_tbl.Properties.VariableNames;
            keep = ~(startsWith(vars, 'mean_') | startsWith(vars, 'std_'));
            reward_diff_summary_table = base_tbl(:, keep);
            % Loop through mean_ columns only
            mean_vars = vars(startsWith(vars, 'mean_'));
            for v = mean_vars
                col = v{1};
                % Stack column across all 3 tables
                vals_mat = cell2mat(cellfun(@(t) t.(col), samp_reward_diff_summary_table, 'UniformOutput', false));
                
                % Compute mean and std across columns (samples)
                reward_diff_summary_table.(col) = mean(vals_mat, 2, 'omitnan');
                % Calculate std of sampled means
                % Create std_ column name by replacing mean_ with std_
                std_col = ['std_' extractAfter(col, 'mean_')];
                reward_diff_summary_table.(std_col) = std(vals_mat, 0, 2, 'omitnan');
            end
            % Assign results to simmed_model_output
            simmed_model_output{param_set_idx,2} = reward_diff_summary_table;
            simmed_model_output{param_set_idx,3} = choice_num_summary_table;
        end
    end
    % Make plots
    % If not doing a parameter sweep, make all plots
    if isempty(MDP.param_to_sweep) 
        make_plots_model_statistics(simmed_model_output{1,2},simmed_model_output{1,3});
    else
        make_plots_param_sweep(simmed_model_output);
    end


end



