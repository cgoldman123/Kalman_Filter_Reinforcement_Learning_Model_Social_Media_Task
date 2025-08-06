function [output_table] = Social_wrapper(varargin)
    close all;
    dbstop if error;
    clearvars -except varargin

    EMPIRICAL = 1; % Indicate if using empirical choices (1) or simulated choices (0).
    % Using empirical choices!
    if EMPIRICAL
        MDP.do_model_free = 1; % Toggle on to do model-free analyses on empirical data.
        MDP.fit_model = 1; % Toggle on to fit the model to empirical data.
        % If fitting the model
        if MDP.fit_model
            MDP.do_simulated_model_free = 1; % Toggle on to do model-free analyses on data simulated using posterior parameter estimates of model.
            MDP.plot_fitted_behavior = 1; % Toggle on to plot behavior after model fitting.
            MDP.save_trial_by_trial_output = 1;  % Toggle on to save trial by trial model latents (e.g., prediction errors) and behavioral data (e.g., reaction times).
        end
    else
        % Using simulated choices!
        MDP.num_samples_to_draw_from_pdf = 0;   %If 0, the model will simulate a choice/RT based on the maximum of the simulated pdf. If >0, it will sample from the distribution of choices/RTs this many times. Note this only matters for models that generate RTs.
        MDP.do_plot_model_statistics = 1; % Toggle on to plot statistics under the current parameter set
        MDP.do_simulated_model_free = 1; % Toggle on to calculate and save model-free analyses on data simulated using parameters set in this main file.
        MDP.do_plot_choice_given_gen_mean = 1; % Toggle on to plot simulated behavior for games of a specific generative mean and horizon (specified below).
        % If plotting simulated data, decide if doing parameter sweep.
        if MDP.do_plot_model_statistics         
            MDP.param_to_sweep = ''; % Specify the name of the parameter name to sweep over or leave this empty to not sweep.
            MDP.param_values_to_sweep_over = linspace(-20, 20, 5); % Specify the values of the parameter to sweep over
        end
        % If plotting simulated data for a given game type
        if MDP.do_plot_choice_given_gen_mean
            gen_mean_difference = 4; % choose a generative mean difference of 2, 4, 8, 12, 24
            horizon = 5; % choose horizon of 1 or 5
            truncate_big_hor = 1; % if truncate_big_hor is true, use the H5 bandit schedule but truncate so that all games are H1
        end
    end
    rng(23);
    
    % If running this code locally
    if ispc
        root = 'L:/';
        experiment = 'prolific'; % indicate local or prolific
        results_dir = sprintf([root 'rsmith/lab-members/cgoldman/Wellbeing/social_media/output/test/']); % Specify directory to save results
        % Specify the subject and room type (like or dislike) to fit, or
        % instead use the forced choices in that behavioral file for
        % simulated data
        if strcmp(experiment,'prolific')
            id = '568d0641b5a2c2000cb657d0';
        elseif strcmp(experiment,'local')
            id = 'AV841';
        end
        room = 'Like';
        % Indicate the model to fit or simulate
        model = "KF_SIGMA"; % Possible models: 'KF_SIGMA_logistic','KF_SIGMA_logistic_DDM', 'KF_SIGMA_logistic_RACING','KF_SIGMA', 'KF_SIGMA_DDM', 'KF_SIGMA_RACING', 'obs_means_logistic', 'obs_means_logistic_DDM'
        MDP.field = {'cong_base_info_bonus'}; % Determine which parameters to fit
    
    % If running this code on the analysis cluster, read in variables using getenv() 
    elseif isunix
        root = '/media/labs/'
        results_dir = getenv('RESULTS')   
        room = getenv('ROOM') 
        experiment = getenv('EXPERIMENT')
        id = getenv('ID')
        model = getenv('MODEL')
        MDP.field = strsplit(getenv('FIELD'), ',')
    end

    
    % Add libraries
    addpath(['./SPM_models/']);
    addpath(['./racing_accumulator/']);
    addpath(['./plotting/']);
    addpath(['./data_processing/']);
    addpath([root '/rsmith/all-studies/util/spm12/']);
    addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);


    model_functions = containers.Map(...
        {'KF_SIGMA_logistic','KF_SIGMA_logistic_DDM', 'KF_SIGMA_logistic_RACING', ...
        'KF_SIGMA', 'KF_SIGMA_DDM', 'KF_SIGMA_RACING',...
        'obs_means_logistic', 'obs_means_logistic_DDM'}, ...
        {@model_SM_KF_SIGMA_logistic, @model_SM_KF_SIGMA_logistic_DDM, @model_SM_KF_SIGMA_logistic_RACING,...
         @model_SM_KF_SIGMA,  @model_SM_KF_SIGMA_DDM, @model_SM_KF_SIGMA_RACING, ...
         @model_SM_obs_means_logistic, @model_SM_obs_means_logistic_DDM});

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
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You should not have to edit anything beyond this point, apart from the model scripts in ./SPM_models

    % display the MDP.params
    disp(MDP.params)
        
    [raw_data,subject_data_info] = get_raw_data(root,experiment,room,id);
    processed_data = process_behavioral_data_SM(raw_data);

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
    

