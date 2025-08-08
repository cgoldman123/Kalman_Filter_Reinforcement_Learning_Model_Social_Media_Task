function [output_table] = Social_wrapper()
    close all;
    dbstop if error;
    clearvars -except varargin

    EMPIRICAL = 1; % Indicate if analyzing empirical choices (1) or simulated choices (0).
    % Using empirical choices!
    if EMPIRICAL
        MDP.do_model_free = 0; % Toggle on to do model-free analyses on empirical data.
        MDP.fit_model = 1; % Toggle on to fit the model to empirical data.
        % If fitting the model
        if MDP.fit_model
            MDP.do_simulated_model_free = 1; % Toggle on to do model-free analyses on data simulated using posterior parameter estimates of model.
            MDP.plot_fitted_behavior = 1; % Toggle on to plot behavior after model fitting.
            MDP.save_trial_by_trial_output = 1;  % Toggle on to save trial by trial model latents (e.g., prediction errors) and behavioral data (e.g., reaction times).
        end
    else
        % Using simulated choices!
        MDP.do_plot_model_statistics = 1; % Toggle on to plot statistics under the current parameter set
        MDP.do_simulated_model_free = 1; % Toggle on to calculate and save model-free analyses on data simulated using parameters set in this main file.
        MDP.do_plot_choice_given_gen_mean = 1; % Toggle on to plot simulated behavior for games of a specific generative mean and horizon (specified below).
        % If plotting simulated data, decide if doing parameter sweep.
        if MDP.do_plot_model_statistics         
            MDP.param_to_sweep = 'side_bias_big_hor'; % Specify the name of the parameter name to sweep over or leave this empty to not sweep.
            MDP.param_values_to_sweep_over = linspace(-20, 20, 5); % Specify the values of the parameter to sweep over
        end
        % If plotting simulated data for a given game type
        if MDP.do_plot_choice_given_gen_mean
            gen_mean_difference = 4; % choose a generative mean difference
            horizon = 5; % choose horizon of 1 or 5
            truncate_big_hor = 1; % if truncate_big_hor is true, use the big bandit schedule but truncate so that all games are H1
        end
    end
    MDP.num_samples_to_draw_from_pdf = 0;   %If 0, the model will simulate a choice/RT based on the maximum of the simulated pdf. If >0, it will sample from the distribution of choices/RTs this many times. Note this only matters for models that generate RTs.

    rng(23);
    
    % If running this code locally
    if ispc || ismac
        if ispc; root = 'L:/';end
        if ismac; root = '/Volumes/labs/';end
        
        study_info.study = 'adm'; % 'wellbeing', 'exercise', 'cobre_neut', 'adm', 'eit'

        %%%%% Specify the data to process for the wellbeing study
        if strcmp(study_info.study,'wellbeing')
            study_info.experiment = 'local'; % indicate local or prolific
            if strcmp(study_info.experiment,'prolific')
                study_info.id = '568d0641b5a2c2000cb657d0';
            elseif strcmp(study_info.experiment,'local')
                study_info.id = 'AV841';
            end
            study_info.room = 'Like';
            results_dir = sprintf([root 'rsmith/lab-members/cgoldman/Wellbeing/social_media/output/test/']); % Specify directory to save results
            addpath('./data_processing/wellbeing_data_processing/');

        %%%% Specify the data to process for the exercise study
        elseif strcmp(study_info.study,'exercise')
            study_info.id = 'AK465'; 
            study_info.run = 'T1';% specify T1, T2, T3, or T4
            study_info.room = 'Like';
            addpath('./data_processing/exercise_study_data_processing/');
            results_dir = sprintf([root 'rsmith/lab-members/cgoldman/savitz_exercise_study/results/']); % Specify directory to save results

        %%%% Specify the data to process for the cobre_neut study
        elseif strcmp(study_info.study,'cobre_neut')
            study_info.id = 'CA038'; 
            study_info.room = 'Like';
            addpath('./data_processing/cobre_neut_data_processing/');
            results_dir = sprintf([root 'rsmith/lab-members/cgoldman/Berg_horizon_task/results/']); % Specify directory to save results

        %%%% Specify the data to process for the adm study
        elseif strcmp(study_info.study,'adm')
            study_info.id = 'AA022'; 
            study_info.condition = 'unloaded';% specify loaded or unloaded
            addpath('./data_processing/adm_data_processing/');
            results_dir = sprintf([root 'rsmith/lab-members/cgoldman/adm/horizon/updated_modeling_results/test/']); % Specify directory to save results
        
        %%%% Specify the data to process for the eit study
        elseif strcmp(study_info.study,'eit')
            study_info.id = 'sub1';
            addpath('./data_processing/EIT_data_processing/');
            results_dir = sprintf([root 'rsmith/lab-members/cgoldman/EIT_horizon/output/test/']); % Specify directory to save results
        end

        % Fill in missing study_info variables with ''
        fields_to_check = {'experiment', 'condition', 'room', 'run'};
        for i = 1:numel(fields_to_check)
            if ~isfield(study_info, fields_to_check{i})
                study_info.(fields_to_check{i}) = '';
            end
        end

        
        % Indicate the model to fit or simulate
        model = "KF_SIGMA"; % Possible models: 'KF_SIGMA_logistic','KF_SIGMA_logistic_DDM', 'KF_SIGMA_logistic_RACING','KF_SIGMA', 'KF_SIGMA_DDM', 'KF_SIGMA_RACING', 'obs_means_logistic', 'obs_means_logistic_DDM', 'obs_means_logistic_RACING'
        MDP.field = {'side_bias'}; % Determine which parameters to fit
    
    % If running this code on the analysis cluster, read in variables using getenv() 
    elseif isunix
        root = '/media/labs/'
        results_dir = getenv('RESULTS')   
        study_info.room = getenv('ROOM') 
        study_info.experiment = getenv('EXPERIMENT')
        study_info.condition = getenv('CONDITION')
        study_info.run = getenv('RUN')
        study_info.id = getenv('ID')
        model = getenv('MODEL')
        MDP.field = strsplit(getenv('FIELD'), ',')
    end
    
    % Add libraries
    addpath('./SPM_models/');
    addpath('./data_processing/');
    addpath('./racing_accumulator/');
    addpath('./plotting/');
    addpath([root '/rsmith/all-studies/util/spm12/']);
    addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);


    model_functions = containers.Map(...
        {'KF_SIGMA_logistic','KF_SIGMA_logistic_DDM', 'KF_SIGMA_logistic_RACING', ...
        'KF_SIGMA', 'KF_SIGMA_DDM', 'KF_SIGMA_RACING',...
        'obs_means_logistic', 'obs_means_logistic_DDM', 'obs_means_logistic_RACING'}, ...
        {@model_SM_KF_SIGMA_logistic, @model_SM_KF_SIGMA_logistic_DDM, @model_SM_KF_SIGMA_logistic_RACING,...
         @model_SM_KF_SIGMA,  @model_SM_KF_SIGMA_DDM, @model_SM_KF_SIGMA_RACING, ...
         @model_SM_obs_means_logistic, @model_SM_obs_means_logistic_DDM, @model_SM_obs_means_logistic_RACING});

    if isKey(model_functions, model)
        MDP.model = model_functions(model);
    else
        error('Unknown model specified in MDP.model');
    end


    %%% Specify the prior parameter parameter values for fitting a model or
    %%% alternatively the parameter values to simulate data.

    if strcmp(model, 'KF_SIGMA_DDM')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'side_bias', 0, 'baseline_noise', 5, ...
            'cong_base_info_bonus', 0, 'incong_base_info_bonus', 0, 'cong_directed_exp', 0, 'incong_directed_exp', 0, ...
            'random_exp', 5, 'sigma_r', 8, 'sigma_d',0,'initial_sigma', 10000, 'decision_thresh_baseline', 3, 'rdiff_bias_mod', 0.05);
        MDP.max_rt = 7;
        MDP.num_choices_to_fit = 5; % fit/sim first free choice (1) or all choices (5)
    elseif strcmp(model, 'KF_SIGMA')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'side_bias', 0, 'baseline_noise', 5, ...
            'cong_base_info_bonus', 0, 'incong_base_info_bonus', 0, 'cong_directed_exp', 0, 'incong_directed_exp', 0, ...
            'random_exp', 5, 'sigma_r', 8, 'sigma_d',0,'initial_sigma', 10000);
        MDP.num_choices_to_fit = 5; % fit/sim first free choice (1) or all choices (5)

    elseif strcmp(model, 'KF_SIGMA_RACING')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'side_bias', 0, 'baseline_noise', 5, ...
            'cong_base_info_bonus', 0, 'incong_base_info_bonus', 0, 'cong_directed_exp', 0, 'incong_directed_exp', 0, ...
            'random_exp', 5, 'sigma_r', 8, 'sigma_d',0,'initial_sigma', 10000, 'decision_thresh_baseline', 2, ...
            'starting_bias_baseline', 0.5, 'drift_baseline', 0, 'drift_reward_diff_mod', 0.1, ...
            'starting_bias_reward_diff_mod', 0.1, 'wd', 0.05, 'ws', 0.05, 'V0', 0);
        MDP.max_rt = 7;
        MDP.num_choices_to_fit = 5; % fit/sim first free choice (1) or all choices (5)

    elseif strcmp(model, 'KF_SIGMA_logistic')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'sigma_r', 8, 'sigma_d',0,'initial_sigma', 10000, ...
            'info_bonus_small_hor', 0, 'info_bonus_big_hor', 0, 'dec_noise_small_hor', 1, ...
            'dec_noise_big_hor', 1, 'side_bias_small_hor', 0, 'side_bias_big_hor', 0);
        MDP.num_choices_to_fit = 1; % DO NOT EDIT. This model is for the first free choice only.

    elseif strcmp(model, 'KF_SIGMA_logistic_DDM')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'sigma_r', 8, 'sigma_d',0,'initial_sigma', 10000, ...
            'info_bonus_small_hor', 0, 'info_bonus_big_hor', 0, 'dec_noise_small_hor', 1, ...
            'dec_noise_big_hor', 1, 'side_bias_small_hor', 0, 'side_bias_big_hor', 0, ...
            'decision_thresh_baseline', 2, 'starting_bias_baseline', 0.5, 'drift_baseline', 0, ...
            'drift_reward_diff_mod', 0.1, 'starting_bias_reward_diff_mod', 0.1);
        MDP.max_rt = 7;
        MDP.num_choices_to_fit = 1; % DO NOT EDIT. This model is for the first free choice only.

    elseif strcmp(model, 'KF_SIGMA_logistic_RACING')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'sigma_r', 8, 'sigma_d',0,'initial_sigma', 10000, ...
            'info_bonus_small_hor', 0, 'info_bonus_big_hor', 0, 'dec_noise_small_hor', 1, ...
            'dec_noise_big_hor', 1, 'side_bias_small_hor', 0, 'side_bias_big_hor', 0, ...
            'decision_thresh_baseline', 2, 'starting_bias_baseline', 0.5, 'drift_baseline', 0, ...
            'drift_reward_diff_mod', 0.1, 'starting_bias_reward_diff_mod', 0.1, ...
            'wd', 0.05, 'ws', 0.05, 'V0', 0);
        MDP.max_rt = 7;
        MDP.num_choices_to_fit = 1; % DO NOT EDIT. This model is for the first free choice only.

    elseif strcmp(model, 'obs_means_logistic')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'info_bonus_small_hor', 0, ...
            'info_bonus_big_hor', 0, 'dec_noise_small_hor', 1, 'dec_noise_big_hor', 1, ...
            'side_bias_small_hor', 0, 'side_bias_big_hor', 0);
        MDP.num_choices_to_fit = 1; % DO NOT EDIT. This model is for the first free choice only.

    elseif strcmp(model, 'obs_means_logistic_DDM')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'info_bonus_small_hor', 0, ...
            'info_bonus_big_hor', 0, 'dec_noise_small_hor', 1, 'dec_noise_big_hor', 1, ...
            'side_bias_small_hor', 0, 'side_bias_big_hor', 0, ...
            'decision_thresh_baseline', 2, 'starting_bias_baseline', 0.5, 'drift_baseline', 0, ...
            'drift_reward_diff_mod', 0.1, 'starting_bias_reward_diff_mod', 0.1);
        MDP.max_rt = 7;
        MDP.num_choices_to_fit = 1; % DO NOT EDIT. This model is for the first free choice only.
    elseif strcmp(model, 'obs_means_logistic_RACING')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'info_bonus_small_hor', 0, ...
            'info_bonus_big_hor', 0, 'dec_noise_small_hor', 1, 'dec_noise_big_hor', 1, ...
            'side_bias_small_hor', 0, 'side_bias_big_hor', 0, ...
            'decision_thresh_baseline', 2, 'starting_bias_baseline', 0.5, 'drift_baseline', 0, ...
            'drift_reward_diff_mod', 0.1, 'starting_bias_reward_diff_mod', 0.1,'wd', 0.05, 'ws', 0.05, 'V0', 0);
        MDP.max_rt = 7;
        MDP.num_choices_to_fit = 1; % DO NOT EDIT. This model is for the first free choice only.
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You should not have to edit anything beyond this point, apart from the model scripts in ./SPM_models

    % display the MDP.params
    disp(MDP.params)


    [processed_data,raw_data,subject_data_info] =  process_data_across_studies(root, study_info);


    % Getting free choices from empirical data
    if EMPIRICAL
        output_table = get_fits(root, processed_data, subject_data_info, results_dir,MDP);
    else
       % Getting free choices from simulated data
        if MDP.do_plot_choice_given_gen_mean & isempty(MDP.param_to_sweep)
            plot_choice_given_gen_mean(processed_data, MDP, gen_mean_difference, horizon, truncate_big_hor);
        end
        if MDP.do_plot_model_statistics
            main_plot_model_stats_or_sweep(processed_data, MDP);
        end
        % Do model free analyses on simulated data
        if MDP.do_simulated_model_free
            output_table = get_simulated_model_free(processed_data, MDP,subject_data_info,root,results_dir);
        end
    end
    

