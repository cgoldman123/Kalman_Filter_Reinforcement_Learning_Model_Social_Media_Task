function [fits, model_output] = fit_extended_model(formatted_file, result_dir, MDP)
    if ispc
        root = 'L:/';
    else
        root = '/media/labs/';
    end
    fprintf('Using this formatted_file: %s\n',formatted_file);

    %formatted_file = 'L:\rsmith\wellbeing\tasks\SocialMedia\output\prolific\kf\beh_Dislike_06_03_24_T16-03-52.csv';  %% remember to comment out
    addpath([root 'rsmith/lab-members/cgoldman/general/']);
    %addpath('~/Documents/MATLAB/MatJAGS/');


    
    fundir      = pwd;%[maindir 'TMS_code/'];
    datadir     = pwd;%[maindir 'TMS_code/'];
    savedir     = pwd;%[maindir];
    addpath(fundir);
    cd(fundir);
    %defaultPlotParameters

%     sub = load_TMS_v1([datadir '/EIT_HorizonTaskOutput_HierarchicalModelFormat_v2.csv']);
    sub = load_TMS_v1(formatted_file);
%     sub = sub(1,:); % REMEMBER TO COMMENT OUT
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
        if ismember(field{i},{'alpha_start', 'alpha_inf', 'associability_weight','noise_learning_rate', 'learning_rate_pos',...
                'learning_rate_neg', 'learning_rate', 'starting_bias', 'drift_mod', 'bias_mod' })
            fits.(field{i}) = 1/(1+exp(-DCM.Ep.(field{i})));
        elseif ismember(field{i},{'dec_noise_h1_13', 'dec_noise_h5_13', 'outcome_informativeness', 'sigma_d', ...
                'info_bonus', 'random_exp', 'initial_sigma_r', 'initial_sigma', 'initial_mu', 'baseline_noise',...
                'sigma_r', 'reward_sensitivity', 'DE_RE_horizon', 'initial_associability', 'decision_thresh'})
            fits.(field{i}) = exp(DCM.Ep.(field{i}));
        elseif ismember(field{i},{'info_bonus_h1', 'info_bonus_h5','side_bias_h1', 'side_bias_h5', 'side_bias',...
                'baseline_info_bonus', 'drift_baseline', 'drift'})
            fits.(field{i}) = DCM.Ep.(field{i});
        elseif any(strcmp(field{i},{'nondecision_time'}))
            fits.(field{i}) = 0.1 + (0.3 - 0.1) ./ (1 + exp(-DCM.Ep.(field{i})));  
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
    fits.average_action_prob = mean(model_output.action_probs(~isnan(model_output.action_probs)), 'all');
    fits.model_acc = sum(model_output.action_probs(~isnan(model_output.action_probs)) > 0.5) / numel(model_output.action_probs(~isnan(model_output.action_probs)));
    fits.F = DCM.F;
    
                
    % simulate behavior with fitted params
    simmed_model_output = MDP.model(fits,actions, rewards,mdp, 1);    

    datastruct.actions = simmed_model_output.actions;
    datastruct.rewards = simmed_model_output.rewards;
    MDP.datastruct = datastruct;
    % note old social media model model_KFcond_v2_SMT
    fprintf( 'Running VB to fit simulated behavior! \n' );

    simfit_DCM = SM_inversion(MDP);

    for i = 1:length(field)
        if ismember(field{i},{'alpha_start', 'alpha_inf', 'associability_weight','noise_learning_rate', 'learning_rate_pos',...
                'learning_rate_neg', 'learning_rate', 'starting_bias', 'drift_mod', 'bias_mod'})
            fits.(['simfit_' field{i}]) = 1/(1+exp(-simfit_DCM.Ep.(field{i})));
        elseif ismember(field{i},{'dec_noise_h1_13', 'dec_noise_h5_13', 'info_bonus', 'outcome_informativeness',...
                'sigma_d', 'info_bonus', 'random_exp','initial_sigma_r', 'initial_sigma', 'initial_mu', 'baseline_noise',...
                'sigma_r', 'reward_sensitivity', 'DE_RE_horizon', 'initial_associability', 'decision_thresh'})
            fits.(['simfit_' field{i}]) = exp(simfit_DCM.Ep.(field{i}));
        elseif ismember(field{i},{'info_bonus_h1', 'info_bonus_h5','side_bias_h1', 'side_bias_h5','side_bias',...
                'baseline_info_bonus', 'drift_baseline', 'drift'})
            fits.(['simfit_' field{i}]) = simfit_DCM.Ep.(field{i});
        elseif any(strcmp(field{i},{'nondecision_time'}))
            fits.(['simfit_' field{i}]) = 0.1 + (0.3 - 0.1) ./ (1 + exp(-simfit_DCM.Ep.(field{i})));     
        else
            disp(field{i});
            error("Param not propertly transformed");
        end
    end
    
    
 end