function [output_table] = Social_wrapper(varargin)
    close all;
    dbstop if error;
    clearvars -except varargin
    % Simulate (and plot) data under the model OR fit the model to actual
    % data. Only toggle one of these on.
    SIM = 0; % Simulate the model
    FIT = 1; % Fit the model
    if FIT
        MDP.get_processed_behavior_and_dont_fit_model = 0; % Toggle on to extract the rts and other processed behavioral data but not fit the model
        MDP.do_model_free = 1; % Toggle on to do model-free analyses on actual data
        MDP.fit_model = 1; % Toggle on to fit the model
        if MDP.fit_model
            MDP.do_simulated_model_free = 1; % Toggle on to do model-free analyses on data simulated using posterior parameter estimates of model.
            MDP.plot_fitted_behavior = 1; % Toggle on to plot behavior after model fitting
        end
    elseif SIM
        MDP.plot_simulated_data = 1; %Toggle on to plot data simulated by model using parameters set in this main file.
        MDP.do_simulated_model_free = 1; % Toggle on to do model-free analyses on data simulated using parameters set in this main file.
    end
    rng(23);
    
    if ispc
        root = 'L:/';
        experiment = 'prolific'; % indicate local or prolific
        results_dir = sprintf([root 'rsmith/lab-members/cgoldman/Wellbeing/social_media/output/test/']);
        if nargin > 0
            id = varargin{1};
            room = varargin{2};
        else
            if strcmp(experiment,'prolific'); id = '568d0641b5a2c2000cb657d0'; elseif strcmp(experiment,'local'); id = 'AV841';end   % CA336 BO224 562eb896733ea000051638c6 666878a27888fdd27f529c64 
            room = 'Like';
        end
        model = "KF_SIGMA_DDM"; % indicate if 'KF_UCB', 'RL', 'KF_UCB_DDM', 'KF_SIGMA_DDM', 'KF_SIGMA', 'KF_SIGMA_RACING
        %MDP.field ={'dec_noise_small_hor','dec_noise_big_hor','side_bias_small_hor','side_bias_big_hor','info_bonus_big_hor','info_bonus_small_hor','sigma_d', 'sigma_r', 'starting_bias_baseline', 'drift_baseline', 'decision_thresh_baseline', 'drift_reward_diff_mod', 'wd', 'ws', 'V0'}; 
        %MDP.field = {'cong_base_info_bonus','incong_base_info_bonus','cong_directed_exp','incong_directed_exp', 'side_bias','random_exp','sigma_d', 'sigma_r', 'baseline_noise', 'rdiff_bias_mod', 'decision_thresh_baseline','wd', 'ws', 'V0'}; % KF_SIGMA_RACING
        %MDP.field = {'cong_base_info_bonus','incong_base_info_bonus','cong_directed_exp','incong_directed_exp', 'side_bias','random_exp','sigma_d', 'sigma_r', 'baseline_noise', 'rdiff_bias_mod', 'decision_thresh_baseline'}; % KF_SIGMA_DDM
        MDP.field = {'cong_base_info_bonus'};
        MDP.settings.num_choices_to_fit = 5; % Specify the number of choices to fit as first free choice (1) or all choices (5)
    
    elseif isunix
    % If running on the analysis cluster, some parameters will be supplied by 
    % the job submission script -- read those accordingly.
        root = '/media/labs/'
        results_dir = getenv('RESULTS')   % run = 1,2,3
        room = getenv('ROOM') %Like and/or Dislike
        experiment = getenv('EXPERIMENT')
        id = getenv('ID')
        model = getenv('MODEL')
        MDP.field = strsplit(getenv('FIELD'), ',')
        MDP.settings.max_rt = 7;
    end
    
    % Add libraries
    addpath(['./SPM_models/']);
    addpath(['./racing_accumulator/']);
    addpath(['./plotting/']);
    addpath(['./data_processing/']);

    model_functions = containers.Map(...
        {'KF_SIGMA_logistic','KF_SIGMA_logistic_DDM','KF_UCB', 'RL', 'KF_UCB_DDM', 'KF_SIGMA_DDM', 'KF_SIGMA' 'KF_SIGMA_logistic_RACING', 'KF_SIGMA_RACING', 'obs_means_logistic', 'obs_means_logistic_DDM'}, ...
        {@model_SM_KF_SIGMA_logistic, @model_SM_KF_SIGMA_logistic_DDM,@model_SM_KF_all_choices, @model_SM_RL_all_choices, @model_SM_KF_DDM_all_choices, @model_SM_KF_SIGMA_DDM, @model_SM_KF_SIGMA, @model_SM_KF_SIGMA_logistic_RACING, @model_SM_KF_SIGMA_RACING, @model_SM_obs_means_logistic, @model_SM_obs_means_logistic_DDM} ...
    );
    if isKey(model_functions, model)
        MDP.model = model_functions(model);
    else
        error('Unknown model specified in MDP.model');
    end
    
    if strcmp(model, 'KF_SIGMA_DDM')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'side_bias', 0, 'baseline_noise', 5, ...
            'cong_base_info_bonus', 0, 'incong_base_info_bonus', 0, 'cong_directed_exp', 0, 'incong_directed_exp', 0, ...
            'random_exp', 5, 'sigma_r', 8, 'initial_sigma', 10000, 'decision_thresh_baseline', 3, 'rdiff_bias_mod', 0.05);
    
    elseif strcmp(model, 'KF_SIGMA')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'side_bias', 0, 'baseline_noise', 5, ...
            'cong_base_info_bonus', 0, 'incong_base_info_bonus', 0, 'cong_directed_exp', 0, 'incong_directed_exp', 0, ...
            'random_exp', 5, 'sigma_r', 8, 'initial_sigma', 10000);
    
    elseif strcmp(model, 'KF_SIGMA_RACING')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'side_bias', 0, 'baseline_noise', 5, ...
            'cong_base_info_bonus', 0, 'incong_base_info_bonus', 0, 'cong_directed_exp', 0, 'incong_directed_exp', 0, ...
            'random_exp', 5, 'sigma_r', 8, 'initial_sigma', 10000, 'decision_thresh_baseline', 2, ...
            'starting_bias_baseline', 0.5, 'drift_baseline', 0, 'drift_reward_diff_mod', 0.1, ...
            'starting_bias_reward_diff_mod', 0.1, 'wd', 0.05, 'ws', 0.05, 'V0', 0);
    
    elseif strcmp(model, 'KF_SIGMA_logistic')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'sigma_r', 8, 'initial_sigma', 10000, ...
            'info_bonus_small_hor', 0, 'info_bonus_big_hor', 0, 'dec_noise_small_hor', 1, ...
            'dec_noise_big_hor', 1, 'side_bias_small_hor', 0, 'side_bias_big_hor', 0);
    
    elseif strcmp(model, 'KF_SIGMA_logistic_DDM')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'sigma_r', 8, 'initial_sigma', 10000, ...
            'info_bonus_small_hor', 0, 'info_bonus_big_hor', 0, 'dec_noise_small_hor', 1, ...
            'dec_noise_big_hor', 1, 'side_bias_small_hor', 0, 'side_bias_big_hor', 0, ...
            'decision_thresh_baseline', 2, 'starting_bias_baseline', 0.5, 'drift_baseline', 0, ...
            'drift_reward_diff_mod', 0.1, 'starting_bias_reward_diff_mod', 0.1);
    
    elseif strcmp(model, 'KF_SIGMA_logistic_RACING')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'sigma_r', 8, 'initial_sigma', 10000, ...
            'info_bonus_small_hor', 0, 'info_bonus_big_hor', 0, 'dec_noise_small_hor', 1, ...
            'dec_noise_big_hor', 1, 'side_bias_small_hor', 0, 'side_bias_big_hor', 0, ...
            'decision_thresh_baseline', 2, 'starting_bias_baseline', 0.5, 'drift_baseline', 0, ...
            'drift_reward_diff_mod', 0.1, 'starting_bias_reward_diff_mod', 0.1, ...
            'wd', 0.05, 'ws', 0.05, 'V0', 0);
    
    elseif strcmp(model, 'obs_means_logistic')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'info_bonus_small_hor', 0, ...
            'info_bonus_big_hor', 0, 'dec_noise_small_hor', 1, 'dec_noise_big_hor', 1, ...
            'side_bias_small_hor', 0, 'side_bias_big_hor', 0);
    
    elseif strcmp(model, 'obs_means_logistic_DDM')
        MDP.params = struct('reward_sensitivity', 1, 'initial_mu', 50, 'info_bonus_small_hor', 0, ...
            'info_bonus_big_hor', 0, 'dec_noise_small_hor', 1, 'dec_noise_big_hor', 1, ...
            'side_bias_small_hor', 0, 'side_bias_big_hor', 0, ...
            'decision_thresh_baseline', 2, 'starting_bias_baseline', 0.5, 'drift_baseline', 0, ...
            'drift_reward_diff_mod', 0.1, 'starting_bias_reward_diff_mod', 0.1);
    end

    % display the MDP.params
    disp(MDP.params)
        
        
    if SIM
        if MDP.plot_simulated_data
            do_plot_choice_given_gen_mean = 1; % Toggle on to plot choice for a given generative mean
            do_plot_model_statistics = 1; % Toggle on to plot statistics under the current parameter set
            MDP.num_samples_to_draw_from_pdf = 0;   %If 0, the model will simulate a choice/RT based on the maximum of the simulated pdf. If >0, it will sample from the distribution of choices/RTs this many times. Note this only matters for models that generate RTs.
            MDP.param_to_sweep = ''; % e.g., side_bias_small_hor leave empty if don't want to sweep over param
            MDP.param_values_to_sweep_over = linspace(-20, 20, 5); 
            if do_plot_choice_given_gen_mean & isempty(MDP.param_to_sweep)
                gen_mean_difference = 4; % choose generative mean difference of 2, 4, 8, 12, 24
                horizon = 5; % choose horizon of 1 or 5
                truncate_big_hor = 1; % if truncate_big_hor is true, use the H5 bandit schedule but truncate so that all games are H1
                plot_choice_given_gen_mean(root, experiment,room, results_dir, MDP, id, gen_mean_difference, horizon, truncate_big_hor);
            end
            if do_plot_model_statistics
                main_plot_model_stats_or_sweep(root, experiment, room, results_dir,MDP, id);
            end
        end
        % Indicate if you would like to do model-free analyses on
        % simulated data
        if MDP.do_simulated_model_free
            output_table = get_simulated_model_free(root, experiment, room, results_dir,MDP,id);
        end
    end
    
    if FIT
        output_table = get_fits(root, experiment, room, results_dir,MDP, id);
    end
