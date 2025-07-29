function plot_model_statistics(experiment, room, cb,MDP)
    % File to do model free analyses on simulated data
    % Load the mdp variable to get bandit schedule
    load(['./SPM_scripts/social_media_' experiment '_mdp_cb' num2str(cb) '.mat']); 

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
            simmed_model_output{param_set_idx,1,1} = MDP.params; % save the parameters used to simulate behavior
            simmed_model_output{param_set_idx,1,2} = MDP.model(MDP.params, actions_and_rts, MDP.rewards, MDP, 1); % save the behavior
            reward_diff_summary_table = get_stats_by_reward_diff(MDP, simmed_model_output{param_set_idx,1,2});
            choice_num_summary_table = get_stats_by_choice_num(MDP, simmed_model_output{param_set_idx,1,2});
            make_plots_model_statistics(reward_diff_summary_table,choice_num_summary_table);
        else
            for sample_num = 1:MDP.num_samples_to_draw_from_pdf
                simmed_model_output{param_set_idx,sample_num,1} = MDP.params; % save the parameters used to simulate behavior
                simmed_model_output{param_set_idx,sample_num,2} = MDP.model(MDP.params, actions_and_rts, MDP.rewards, MDP, 1); % save the behavior
                reward_diff_summary_table = get_stats_by_reward_diff(MDP, simmed_model_output{param_set_idx,1,2});
                choice_num_summary_table = get_stats_by_choice_num(MDP, simmed_model_output{param_set_idx,1,2});
            end
        end
    end


end



