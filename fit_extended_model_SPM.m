function [fits, model_output] = fit_extended_model_SPM(processed_data, MDP)

    fprintf( 'Running Newton Function to fit\n' );
    MDP.processed_data = processed_data;

    DCM = SM_inversion(MDP);
   

    
    %% Organize fits
    DCM_field = DCM.field;
    % get fitted and fixed params
    fits = DCM.params;
    for i = 1:length(DCM_field)
        if ismember(DCM_field{i},{'learning_rate', 'learning_rate_pos', 'learning_rate_neg', 'noise_learning_rate', 'alpha_start', 'alpha_inf', 'associability_weight', ...
                'starting_bias_baseline', 'ws', 'wd'})
            fits.(DCM_field{i}) = 1/(1+exp(-DCM.Ep.(DCM_field{i})));
        elseif ismember(DCM_field{i},{'dec_noise_small_hor', 'dec_noise_big_hor', 'h5_baseline_dec_noise', 'h5_slope_dec_noise', ...
                'initial_sigma', 'initial_sigma_r', 'initial_mu', 'initial_associability', ...
                'drift_action_prob_mod', 'drift_reward_diff_mod', 'drift_UCB_diff_mod',...
                'starting_bias_action_prob_mod', 'starting_bias_reward_diff_mod', 'starting_bias_UCB_diff_mod',...
                'decision_thresh_action_prob_mod', 'decision_thresh_reward_diff_mod', 'decision_thresh_UCB_diff_mod', 'decision_thresh_decision_noise_mod' ...
                'outcome_informativeness', 'baseline_noise', ...
                'reward_sensitivity', 'DE_RE_horizon'})
            fits.(DCM_field{i}) = exp(DCM.Ep.(DCM_field{i}));
        elseif ismember(DCM_field{i},{'h5_baseline_info_bonus', 'h5_slope_info_bonus', 'info_bonus_small_hor', 'baseline_info_bonus', ...
                'side_bias', 'side_bias_small_hor', 'side_bias_big_hor', 'info_bonus', 'info_bonus_big_hor', 'random_exp', 'rdiff_bias_mod',...
                'drift_baseline', 'drift','directed_exp', 'V0','cong_base_info_bonus','incong_base_info_bonus','cong_directed_exp','incong_directed_exp'})
            fits.(DCM_field{i}) = DCM.Ep.(DCM_field{i});
        elseif any(strcmp(DCM_field{i},{'nondecision_time'}))
            fits.(DCM_field{i}) = 0.1 + (0.3 - 0.1) ./ (1 + exp(-DCM.Ep.(DCM_field{i})));  
        elseif any(strcmp(DCM_field{i},{'decision_thresh_baseline'}))
            fits.(DCM_field{i}) = .5 + (1000 - .5) ./ (1 + exp(-DCM.Ep.(DCM_field{i}))); 
        elseif any(strcmp(DCM_field{i},{'sigma_d', 'sigma_r'}))
            fits.(DCM_field{i}) = (40) ./ (1 + exp(-DCM.Ep.(DCM_field{i}))); 
        else
            disp(DCM_field{i});
            error("Param not propertly transformed");
        end
    end
    
    
    actions_and_rts.actions = processed_data.actions;
    actions_and_rts.RTs = processed_data.RTs;
    rewards = processed_data.rewards;
    
    model_output = MDP.model(fits,actions_and_rts, rewards,MDP, 0);    
    
    % Fill in RTs for the plots for models that are choice only
    if all(all(isnan(model_output.rts)))
        model_output.rts = processed_data.RTs;
    end

    model_output.DCM = DCM;
    fits.average_action_prob = mean(model_output.action_probs(~isnan(model_output.action_probs)), 'all');
    
    fits.average_action_prob_small_hor_1 = mean(model_output.action_probs(find(processed_data.horizon_type==1),5), 'omitnan');
    
    % Dynamically assign average action prob for big horizon games
    for i = 1:processed_data.num_free_choices_big_hor
        col_idx = i + 4; % skip over the forced choices
        fieldname = sprintf('average_action_prob_big_hor_%d', i);
        fits.(fieldname) = mean(model_output.action_probs(find(processed_data.horizon_type==2), col_idx), 'omitnan');
    end

    fits.model_acc = sum(model_output.action_probs(~isnan(model_output.action_probs)) > 0.5) / numel(model_output.action_probs(~isnan(model_output.action_probs)));
    fits.F = DCM.F;

    % If the model contains a DDM or racing accumulator, save the number of invalid RTs
    model_str = func2str(MDP.model);
    if contains(model_str, 'DDM') || contains(model_str, 'RACING')   
        fits.num_invalid_rts = model_output.num_invalid_rts;
    end

    % Plot the fitted behavior!
    if MDP.plot_fitted_behavior
        reward_diff_summary_table = get_stats_by_reward_diff(MDP, model_output);
        choice_num_summary_table = get_stats_by_choice_num(MDP, model_output);
        make_plots_model_statistics(reward_diff_summary_table,choice_num_summary_table,processed_data);
    end
                
    % simulate behavior with fitted params
    simmed_model_output = MDP.model(fits,actions_and_rts, rewards,MDP, 1);    

    MDP.processed_data.actions = simmed_model_output.actions;
    MDP.processed_data.rewards = simmed_model_output.rewards;
    % If the model contains a DDM or racing accumulator, save simulated RTs
    if contains(model_str, 'DDM') || contains(model_str, 'RACING')   
        MDP.processed_data.RTs = simmed_model_output.rts;
    else
        MDP.processed_data.RTs = nan(MDP.processed_data.num_games,MDP.processed_data.num_free_choices_big_hor+MDP.processed_data.num_forced_choices);
    end


    % note old social media model model_KFcond_v2_SMT
    fprintf( 'Running VB to fit simulated behavior! \n' );

    simfit_DCM = SM_inversion(MDP);
    model_output.simfit_DCM = simfit_DCM;

    for i = 1:length(DCM_field)
        if ismember(DCM_field{i},{'learning_rate', 'learning_rate_pos', 'learning_rate_neg', 'noise_learning_rate', 'alpha_start', 'alpha_inf', 'associability_weight', ...
                'starting_bias_baseline', 'ws', 'wd'})
            fits.(['simfit_' DCM_field{i}]) = 1/(1+exp(-simfit_DCM.Ep.(DCM_field{i})));
        elseif ismember(DCM_field{i},{'dec_noise_small_hor', 'dec_noise_big_hor', 'h5_baseline_dec_noise', 'h5_slope_dec_noise', ...
                'initial_sigma', 'initial_sigma_r', 'initial_mu', 'initial_associability', ...
                'drift_action_prob_mod', 'drift_reward_diff_mod', 'drift_UCB_diff_mod',...
                'starting_bias_action_prob_mod', 'starting_bias_reward_diff_mod', 'starting_bias_UCB_diff_mod',...
                'decision_thresh_action_prob_mod', 'decision_thresh_reward_diff_mod', 'decision_thresh_UCB_diff_mod', 'decision_thresh_decision_noise_mod'...
                'outcome_informativeness', 'baseline_noise',...
                'reward_sensitivity', 'DE_RE_horizon'})
            fits.(['simfit_' DCM_field{i}]) = exp(simfit_DCM.Ep.(DCM_field{i}));
        elseif ismember(DCM_field{i},{'h5_baseline_info_bonus', 'h5_slope_info_bonus', 'info_bonus_small_hor', 'baseline_info_bonus',...
                'side_bias', 'side_bias_small_hor', 'side_bias_big_hor', 'info_bonus', 'info_bonus_big_hor', 'random_exp', 'rdiff_bias_mod',...
                'drift_baseline', 'drift', 'directed_exp', 'V0','cong_base_info_bonus','incong_base_info_bonus','cong_directed_exp','incong_directed_exp'})
            fits.(['simfit_' DCM_field{i}]) = simfit_DCM.Ep.(DCM_field{i});
        elseif any(strcmp(DCM_field{i},{'nondecision_time'}))
            fits.(['simfit_' DCM_field{i}]) = 0.1 + (0.3 - 0.1) ./ (1 + exp(-simfit_DCM.Ep.(DCM_field{i})));     
        elseif any(strcmp(DCM_field{i},{'decision_thresh_baseline'}))
            fits.(['simfit_' DCM_field{i}]) = .5 + (1000 - .5) ./ (1 + exp(-simfit_DCM.Ep.(DCM_field{i})));
        elseif any(strcmp(DCM_field{i},{'sigma_d', 'sigma_r'}))
            fits.(['simfit_' DCM_field{i}]) = (40) ./ (1 + exp(-simfit_DCM.Ep.(DCM_field{i})));
        else 
            disp(DCM_field{i});
            error("Param not propertly transformed");
        end
    end
    
    
 end