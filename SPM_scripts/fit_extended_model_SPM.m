function [fits, model_output] = fit_extended_model_SPM(formatted_file, result_dir, MDP)
    if ispc
        root = 'L:/';
    else
        root = '/media/labs/';
    end
    fprintf('Using this formatted_file: %s\n',formatted_file);

    %formatted_file = 'L:\rsmith\wellbeing\tasks\SocialMedia\output\prolific\kf\beh_Dislike_06_03_24_T16-03-52.csv';  %% remember to comment out
    addpath([root 'rsmith/lab-members/cgoldman/general/']);


    sub = load_TMS_v1(formatted_file);

    % If we are just getting the rts and not fitting the model, return
    if MDP.get_rts_and_dont_fit_model
        fits = sub.RT; % return RT as fits as a placeholder
        model_output.results.RT = sub.RT; % store RTs in model output
        model_output.results.choices = sub.a; % store choices in model output
        return;
    end



    disp(sub);
    
    %% ========================================================================
    %% HIERARCHICAL MODEL FIT ON FIRST FREE CHOICE %%
    %% HIERARCHICAL MODEL FIT ON FIRST FREE CHOICE %%
    %% HIERARCHICAL MODEL FIT ON FIRST FREE CHOICE %%
    %% ========================================================================

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
    field = DCM.field;
    % get fitted and fixed params
    fits = DCM.params;
    for i = 1:length(field)
        if ismember(field{i},{'learning_rate', 'learning_rate_pos', 'learning_rate_neg', 'noise_learning_rate', 'alpha_start', 'alpha_inf', 'associability_weight', ...
                'starting_bias_baseline', 'ws', 'wd'})
            fits.(field{i}) = 1/(1+exp(-DCM.Ep.(field{i})));
        elseif ismember(field{i},{'h1_dec_noise', 'h5_dec_noise', 'h5_baseline_dec_noise', 'h5_slope_dec_noise', ...
                'initial_sigma', 'initial_sigma_r', 'initial_mu', 'initial_associability', ...
                'drift_action_prob_mod', 'drift_reward_diff_mod', 'drift_UCB_diff_mod',...
                'starting_bias_action_prob_mod', 'starting_bias_reward_diff_mod', 'starting_bias_UCB_diff_mod',...
                'decision_thresh_action_prob_mod', 'decision_thresh_reward_diff_mod', 'decision_thresh_UCB_diff_mod', 'decision_thresh_decision_noise_mod' ...
                'outcome_informativeness', 'random_exp', 'baseline_noise', ...
                'reward_sensitivity', 'DE_RE_horizon'})
            fits.(field{i}) = exp(DCM.Ep.(field{i}));
        elseif ismember(field{i},{'h5_baseline_info_bonus', 'h5_slope_info_bonus', 'h1_info_bonus', 'baseline_info_bonus', ...
                'side_bias', 'side_bias_h1', 'side_bias_h5', 'info_bonus', 'h5_info_bonus', ...
                'drift_baseline', 'drift','directed_exp', 'V0'})
            fits.(field{i}) = DCM.Ep.(field{i});
        elseif any(strcmp(field{i},{'nondecision_time'}))
            fits.(field{i}) = 0.1 + (0.3 - 0.1) ./ (1 + exp(-DCM.Ep.(field{i})));  
        elseif any(strcmp(field{i},{'decision_thresh_baseline'}))
            fits.(field{i}) = .5 + (1000 - .5) ./ (1 + exp(-DCM.Ep.(field{i}))); 
        elseif any(strcmp(field{i},{'sigma_d', 'sigma_r'}))
            fits.(field{i}) = (40) ./ (1 + exp(-DCM.Ep.(field{i}))); 
        else
            disp(field{i});
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

    if ismember(func2str(MDP.model), {'model_SM_KF_DDM_all_choices', 'model_SM_KF_SIGMA_DDM_all_choices', 'model_SM_KF_SIGMA_logistic_DDM', 'model_SM_KF_SIGMA_logistic_RACING'})
        fits.num_invalid_rts = model_output.num_invalid_rts;
    end

    
                
    % simulate behavior with fitted params
    simmed_model_output = MDP.model(fits,actions_and_rts, rewards,mdp, 1);    

    datastruct.actions = simmed_model_output.actions;
    datastruct.rewards = simmed_model_output.rewards;
    if ismember(func2str(MDP.model), {'model_SM_KF_DDM_all_choices', 'model_SM_KF_SIGMA_DDM_all_choices', 'model_SM_KF_SIGMA_logistic_DDM', 'model_SM_KF_SIGMA_logistic_RACING'})
        datastruct.RTs = simmed_model_output.rts;
    else
        datastruct.RTs = nan(40,9);
    end


    MDP.datastruct = datastruct;
    % note old social media model model_KFcond_v2_SMT
    fprintf( 'Running VB to fit simulated behavior! \n' );

    simfit_DCM = SM_inversion(MDP);
    model_output.simfit_DCM = simfit_DCM;

    for i = 1:length(field)
        if ismember(field{i},{'learning_rate', 'learning_rate_pos', 'learning_rate_neg', 'noise_learning_rate', 'alpha_start', 'alpha_inf', 'associability_weight', ...
                'starting_bias_baseline', 'ws', 'wd'})
            fits.(['simfit_' field{i}]) = 1/(1+exp(-simfit_DCM.Ep.(field{i})));
        elseif ismember(field{i},{'h1_dec_noise', 'h5_dec_noise', 'h5_baseline_dec_noise', 'h5_slope_dec_noise', ...
                'initial_sigma', 'initial_sigma_r', 'initial_mu', 'initial_associability', ...
                'drift_action_prob_mod', 'drift_reward_diff_mod', 'drift_UCB_diff_mod',...
                'starting_bias_action_prob_mod', 'starting_bias_reward_diff_mod', 'starting_bias_UCB_diff_mod',...
                'decision_thresh_action_prob_mod', 'decision_thresh_reward_diff_mod', 'decision_thresh_UCB_diff_mod', 'decision_thresh_decision_noise_mod'...
                'outcome_informativeness',  'random_exp', 'baseline_noise',...
                'reward_sensitivity', 'DE_RE_horizon'})
            fits.(['simfit_' field{i}]) = exp(simfit_DCM.Ep.(field{i}));
        elseif ismember(field{i},{'h5_baseline_info_bonus', 'h5_slope_info_bonus', 'h1_info_bonus', 'baseline_info_bonus',...
                'side_bias', 'side_bias_h1', 'side_bias_h5', 'info_bonus', 'h5_info_bonus',...
                'drift_baseline', 'drift', 'directed_exp', 'V0'})
            fits.(['simfit_' field{i}]) = simfit_DCM.Ep.(field{i});
        elseif any(strcmp(field{i},{'nondecision_time'}))
            fits.(['simfit_' field{i}]) = 0.1 + (0.3 - 0.1) ./ (1 + exp(-simfit_DCM.Ep.(field{i})));     
        elseif any(strcmp(field{i},{'decision_thresh_baseline'}))
            fits.(['simfit_' field{i}]) = .5 + (1000 - .5) ./ (1 + exp(-simfit_DCM.Ep.(field{i})));
        elseif any(strcmp(field{i},{'sigma_d', 'sigma_r'}))
            fits.(['simfit_' field{i}]) = (40) ./ (1 + exp(-simfit_DCM.Ep.(field{i})));
        else 
            disp(field{i});
            error("Param not propertly transformed");
        end
    end
    
    
 end