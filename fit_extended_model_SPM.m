function [fits, model_output] = fit_extended_model_SPM(formatted_file, result_dir, MDP)
    fprintf('Using this formatted_file: %s\n',formatted_file);

    %formatted_file = 'L:\rsmith\wellbeing\tasks\SocialMedia\output\prolific\kf\beh_Dislike_06_03_24_T16-03-52.csv';  %% remember to comment out

    sub = process_behavioral_data_SM(formatted_file);

    disp(sub);
    
    %% prep data structure 
    NS = length(sub);   % number of subjects
    T = 4;              % number of forced choices
    
    NUM_GAMES = 40; %max(vertcat(sub.game), [], 'all');
    

    GL = nan(NS,   NUM_GAMES);

    for sn = 1:length(sub)


        % game length
        dum = sub(sn).gameLength;
        GL(sn,1:size(dum,1)) = dum;

        G(sn) = length(dum);

        % information difference
        dum = sub(sn).uc - 2;
        dI(sn, 1:size(dum,1)) = -dum;


    end

    GL(GL==5) = 1;
    GL(GL==9) = 2; %used to be 10

    C1 = GL ;      %(GL-1)*2+UC;      CAL edits
    nC1 = 2;



    datastruct = struct(...
        'C1', C1, 'nC1', nC1, ...
        'NS', NS, 'G',  G,  'T',   T, ...
        'dI', dI, 'actions',  sub.a,  'RTs', sub.RT, 'rewards', sub.r, 'bandit1_schedule', sub.bandit1_schedule,...
        'bandit2_schedule', sub.bandit2_schedule, 'settings', MDP.settings, 'result_dir', result_dir);
    
    % mdp = datastruct;
    % save('./SPM_scripts/social_media_local_mdp_cb1.mat', 'mdp');
    
    % If we are just getting the rts/datastruct and not fitting the model, return
    if MDP.get_processed_behavior_and_dont_fit_model
        fits = sub.RT; % return RT as fits as a placeholder
        model_output.results.RT = sub.RT; % store RTs in model output
        model_output.results.choices = sub.a; % store choices in model output
        model_output.datastruct = datastruct; % store full datastruct for simulating behavior
        return;
    end

    if ispc
        root = 'L:/';
    elseif ismac
        root = '/Volumes/labs/';
    elseif isunix 
        root = '/media/labs/';
    end
    
    fprintf( 'Running Newton Function to fit\n' );
    MDP.datastruct = datastruct;

    DCM = SM_inversion(MDP);
   

    
    %% Organize fits
    DCM_field = DCM.field;
    % get fitted and fixed params
    fits = DCM.params;
    for i = 1:length(DCM_field)
        if ismember(DCM_field{i},{'learning_rate', 'learning_rate_pos', 'learning_rate_neg', 'noise_learning_rate', 'alpha_start', 'alpha_inf', 'associability_weight', ...
                'starting_bias_baseline', 'ws', 'wd'})
            fits.(DCM_field{i}) = 1/(1+exp(-DCM.Ep.(DCM_field{i})));
        elseif ismember(DCM_field{i},{'h1_dec_noise', 'h5_dec_noise', 'h5_baseline_dec_noise', 'h5_slope_dec_noise', ...
                'initial_sigma', 'initial_sigma_r', 'initial_mu', 'initial_associability', ...
                'drift_action_prob_mod', 'drift_reward_diff_mod', 'drift_UCB_diff_mod',...
                'starting_bias_action_prob_mod', 'starting_bias_reward_diff_mod', 'starting_bias_UCB_diff_mod',...
                'decision_thresh_action_prob_mod', 'decision_thresh_reward_diff_mod', 'decision_thresh_UCB_diff_mod', 'decision_thresh_decision_noise_mod' ...
                'outcome_informativeness', 'baseline_noise', ...
                'reward_sensitivity', 'DE_RE_horizon'})
            fits.(DCM_field{i}) = exp(DCM.Ep.(DCM_field{i}));
        elseif ismember(DCM_field{i},{'h5_baseline_info_bonus', 'h5_slope_info_bonus', 'h1_info_bonus', 'baseline_info_bonus', ...
                'side_bias', 'side_bias_h1', 'side_bias_h5', 'info_bonus', 'h5_info_bonus', 'random_exp', 'rdiff_bias_mod',...
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
    
    
    actions_and_rts.actions = datastruct.actions;
    actions_and_rts.RTs = datastruct.RTs;
    rewards = datastruct.rewards;

    mdp = datastruct;
    % note that mu2 == right bandit ==  c=2 == free choice = 1

    
    
    model_output = MDP.model(fits,actions_and_rts, rewards,mdp, 0);    
    model_output.DCM = DCM;
    fits.average_action_prob = mean(model_output.action_probs(~isnan(model_output.action_probs)), 'all');
    
    fits.average_action_prob_H1_1 = mean(model_output.action_probs(1:2:end, 5), 'omitnan');
    fits.average_action_prob_H5_1 = mean(model_output.action_probs(2:2:end, 5), 'omitnan');
    fits.average_action_prob_H5_2 = mean(model_output.action_probs(2:2:end, 6), 'omitnan');
    fits.average_action_prob_H5_3 = mean(model_output.action_probs(2:2:end, 7), 'omitnan');
    fits.average_action_prob_H5_4 = mean(model_output.action_probs(2:2:end, 8), 'omitnan');
    fits.average_action_prob_H5_5 = mean(model_output.action_probs(2:2:end, 9), 'omitnan');
    fits.model_acc = sum(model_output.action_probs(~isnan(model_output.action_probs)) > 0.5) / numel(model_output.action_probs(~isnan(model_output.action_probs)));
    fits.F = DCM.F;

    % If the model contains a DDM or racing accumulator, save the number of invalid RTs
    model_str = func2str(MDP.model);
    if contains(model_str, 'DDM') || contains(model_str, 'RACING')   
        fits.num_invalid_rts = model_output.num_invalid_rts;
    end

    % Plot the fitted behavior!
    if MDP.plot_fitted_behavior
        % First fill in model_output
        datastruct_fields = fieldnames(datastruct);
        for idx = 1:length(datastruct_fields)
            field = datastruct_fields{idx};  % extract the field name string
            MDP.(field) = datastruct.(field);
        end
        reward_diff_summary_table = get_stats_by_reward_diff(MDP, model_output);
        choice_num_summary_table = get_stats_by_choice_num(MDP, model_output);
        make_plots_model_statistics(reward_diff_summary_table,choice_num_summary_table);
    end
                
    % simulate behavior with fitted params
    mdp.num_samples_to_draw_from_pdf = 0;
    simmed_model_output = MDP.model(fits,actions_and_rts, rewards,mdp, 1);    

    datastruct.actions = simmed_model_output.actions;
    datastruct.rewards = simmed_model_output.rewards;
    % If the model contains a DDM or racing accumulator, save simulated RTs
    if contains(model_str, 'DDM') || contains(model_str, 'RACING')   
        datastruct.RTs = simmed_model_output.rts;
    else
        datastruct.RTs = nan(40,9);
    end


    MDP.datastruct = datastruct;
    % note old social media model model_KFcond_v2_SMT
    fprintf( 'Running VB to fit simulated behavior! \n' );

    simfit_DCM = SM_inversion(MDP);
    model_output.simfit_DCM = simfit_DCM;

    for i = 1:length(DCM_field)
        if ismember(DCM_field{i},{'learning_rate', 'learning_rate_pos', 'learning_rate_neg', 'noise_learning_rate', 'alpha_start', 'alpha_inf', 'associability_weight', ...
                'starting_bias_baseline', 'ws', 'wd'})
            fits.(['simfit_' DCM_field{i}]) = 1/(1+exp(-simfit_DCM.Ep.(DCM_field{i})));
        elseif ismember(DCM_field{i},{'h1_dec_noise', 'h5_dec_noise', 'h5_baseline_dec_noise', 'h5_slope_dec_noise', ...
                'initial_sigma', 'initial_sigma_r', 'initial_mu', 'initial_associability', ...
                'drift_action_prob_mod', 'drift_reward_diff_mod', 'drift_UCB_diff_mod',...
                'starting_bias_action_prob_mod', 'starting_bias_reward_diff_mod', 'starting_bias_UCB_diff_mod',...
                'decision_thresh_action_prob_mod', 'decision_thresh_reward_diff_mod', 'decision_thresh_UCB_diff_mod', 'decision_thresh_decision_noise_mod'...
                'outcome_informativeness', 'baseline_noise',...
                'reward_sensitivity', 'DE_RE_horizon'})
            fits.(['simfit_' DCM_field{i}]) = exp(simfit_DCM.Ep.(DCM_field{i}));
        elseif ismember(DCM_field{i},{'h5_baseline_info_bonus', 'h5_slope_info_bonus', 'h1_info_bonus', 'baseline_info_bonus',...
                'side_bias', 'side_bias_h1', 'side_bias_h5', 'info_bonus', 'h5_info_bonus', 'random_exp', 'rdiff_bias_mod',...
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