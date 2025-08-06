function [output_table] = Social_wrapper(varargin)
    %% If running the pyddm scripts for the first time, make sure the correct python virtual environment is called. Run this code to activate the correct virtual environment
    % pyenv('Version', 'C:\Users\CGoldman\AppData\Local\anaconda3\envs\pyddm\python.exe')
    close all;
    dbstop if error;
    %% Clear workspace
    clearvars -except varargin
    % Simulate (and plot) data under the model OR fit the model to actual
    % data. Only toggle one of these on.
    SIM = 1; % Simulate the model
    FIT = 0; % Fit the model
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
    
    % warning('off', 'all');
    %% Construct the appropriate path depending on the system this is run on
    % If running on the analysis cluster, some parameters will be supplied by 
    % the job submission script -- read those accordingly.
    
    if ispc
        fitting_procedure = "SPM"; % Specify fitting procedure as "SPM", "VBA", or "PYDDM"
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
        % Only set the anonymous function for the model and assign the
        % field/drift mappings if NOT fitting with PYDDM. If you are
        % fitting with PYDDM, you'll have to set the bounds for each parameter
        % in the fitting file.
        if strcmp(fitting_procedure, "PYDDM")
            MDP.settings = 'fit all choices and rts'; %(e.g., "fit all choices and rts", "fit first free choice and rt")
        elseif ~strcmp(fitting_procedure, "PYDDM")
            model = "KF_SIGMA_DDM"; % indicate if 'KF_UCB', 'RL', 'KF_UCB_DDM', 'KF_SIGMA_DDM', 'KF_SIGMA', 'KF_SIGMA_RACING
            %MDP.field ={'dec_noise_small_hor','dec_noise_big_hor','side_bias_small_hor','side_bias_big_hor','info_bonus_big_hor','info_bonus_small_hor','sigma_d', 'sigma_r', 'starting_bias_baseline', 'drift_baseline', 'decision_thresh_baseline', 'drift_reward_diff_mod', 'wd', 'ws', 'V0'}; 
            %MDP.field = {'cong_base_info_bonus','incong_base_info_bonus','cong_directed_exp','incong_directed_exp', 'side_bias','random_exp','sigma_d', 'sigma_r', 'baseline_noise', 'rdiff_bias_mod', 'decision_thresh_baseline','wd', 'ws', 'V0'}; % KF_SIGMA_RACING
            %MDP.field = {'cong_base_info_bonus','incong_base_info_bonus','cong_directed_exp','incong_directed_exp', 'side_bias','random_exp','sigma_d', 'sigma_r', 'baseline_noise', 'rdiff_bias_mod', 'decision_thresh_baseline'}; % KF_SIGMA_DDM
            MDP.field = {'cong_base_info_bonus'};
            if strcmp(fitting_procedure, "VBA")
                MDP.observation_params = MDP.field; % When there is no latent state learning, all params are observation params
            end
            MDP.settings.num_choices_to_fit = 5; % Specify the number of choices to fit as first free choice (1) or all choices (5)
            if contains(model, 'DDM') || contains(model, 'RACING')     
                % possible mappings are action_prob, reward_diff, UCB,
                % side_bias, decsision_noise
                MDP.settings.drift_mapping = {''};
                MDP.settings.bias_mapping = {''};
                MDP.settings.thresh_mapping = {''};
                MDP.settings.max_rt = 7;
            end
        end
        
    elseif isunix
        fitting_procedure = getenv('FITTING_PROCEDURE')
        root = '/media/labs/'
        results_dir = getenv('RESULTS')   % run = 1,2,3
        room = getenv('ROOM') %Like and/or Dislike
        experiment = getenv('EXPERIMENT')
        id = getenv('ID')

        % Only set the anonymous function for the model and assign the
        % field/drift mappings if NOT fitting with PYDDM. If you are
        % fitting with PYDDM, you'll have to set the bounds for each parameter
        % in the fitting file.
        if strcmp(fitting_procedure, "PYDDM")
            MDP.settings = getenv('FIELD'); % The FIELD variable from the environment is used to communicate settings in the PYDDM
            
        elseif ~strcmp(fitting_procedure, "PYDDM")
            model = getenv('MODEL')
            MDP.field = strsplit(getenv('FIELD'), ',')
            if strcmp(fitting_procedure, "VBA")
                MDP.observation_params = MDP.field;
            end
    
            if ismember(model, {'KF_SIGMA_DDM'})
                % Set up drift mapping
                MDP.settings.drift_mapping = strsplit(getenv('DRIFT_MAPPING'), ',');
                if  strcmp(MDP.settings.drift_mapping{1}, 'none')
                    MDP.settings.drift_mapping = {};
                end
                % Set up bias mapping
                MDP.settings.bias_mapping = strsplit(getenv('BIAS_MAPPING'), ',');
                if strcmp(MDP.settings.bias_mapping{1}, 'none')
                    MDP.settings.bias_mapping = {};
                end
                % Set up threshold mapping  
                MDP.settings.thresh_mapping = strsplit(getenv('THRESH_MAPPING'), ',');
                if strcmp(MDP.settings.thresh_mapping{1}, 'none')
                    MDP.settings.thresh_mapping = {};
                end
                fprintf('Drift mappings: %s\n', strjoin(MDP.settings.drift_mapping));
                fprintf('Bias mappings: %s\n', strjoin(MDP.settings.bias_mapping));
                fprintf('Threshold mappings: %s\n', strjoin(MDP.settings.thresh_mapping));
                MDP.settings.max_rt = 7;
            else
                MDP.settings = '';
            end
        end
    end
    
    % Add libraries. Some of these are for the VBA example code and may not
    % be needed.
    addpath(['./SPM_models/']);
    addpath(['./SPM_models/legacy_models/']);
    addpath(['./VBA_scripts/']);
    addpath(['./racing_accumulator/']);
    addpath(['./plotting/']);
    addpath(['./data_processing/']);
    addpath([root '/rsmith/all-studies/util/spm12/']);
    addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);
    addpath([root '/rsmith/all-studies/util/VBA-toolbox-master/']);
    addpath([root '/rsmith/all-studies/util/VBA-toolbox-master/modules/theory_of_mind/']);
    addpath([root '/rsmith/all-studies/util/VBA-toolbox-master/utils/']);
    addpath([root '/rsmith/all-studies/util/VBA-toolbox-master/demos/_models/']);
    addpath([root '/rsmith/all-studies/util/VBA-toolbox-master/core/']);
    addpath([root '/rsmith/all-studies/util/VBA-toolbox-master/core/diagnostics/']);
    addpath([root '/rsmith/all-studies/util/VBA-toolbox-master/core/display/']);
    addpath([root '/rsmith/all-studies/util/VBA-toolbox-master/modules/GLM/']);


    % Only set the anonymous function for the model and assign the
    % prior/fixed parameter values if NOT fitting with PYDDM. If you are
    % fitting with PYDDM, you'll have to set the bounds for each parameter
    % in the fitting file.
    if ~strcmp(fitting_procedure,'PYDDM')
        model_functions = containers.Map(...
            {'KF_SIGMA_logistic','KF_SIGMA_logistic_DDM','KF_UCB', 'RL', 'KF_UCB_DDM', 'KF_SIGMA_DDM', 'KF_SIGMA' 'KF_SIGMA_logistic_RACING', 'KF_SIGMA_RACING', 'obs_means_logistic', 'obs_means_logistic_DDM'}, ...
            {@model_SM_KF_SIGMA_logistic, @model_SM_KF_SIGMA_logistic_DDM,@model_SM_KF_all_choices, @model_SM_RL_all_choices, @model_SM_KF_DDM_all_choices, @model_SM_KF_SIGMA_DDM, @model_SM_KF_SIGMA, @model_SM_KF_SIGMA_logistic_RACING, @model_SM_KF_SIGMA_RACING, @model_SM_obs_means_logistic, @model_SM_obs_means_logistic_DDM} ...
        );
        if isKey(model_functions, model)
            MDP.model = model_functions(model);
        else
            error('Unknown model specified in MDP.model');
        end
    
        % Parameters fixed or fit in all models
        MDP.params.reward_sensitivity = 1;
        MDP.params.initial_mu = 50;
    

        % Parameters fixed or fit in certain models
        if ismember(model, {'KF_UCB', 'KF_UCB_DDM', 'KF_SIGMA_DDM', 'KF_SIGMA', 'RL', 'KF_SIGMA_RACING'})
            MDP.params.side_bias =  0; 
            MDP.params.baseline_noise = 5; 
            
            if ismember(model, {'KF_UCB', 'KF_UCB_DDM', 'RL'})
                MDP.params.baseline_info_bonus =  0; 
                % make directed exploration and random exploration same param or keep
                % together
                if any(strcmp('DE_RE_horizon', MDP.field))
                    MDP.params.DE_RE_horizon = 2.5; % prior on this value
                else
                    MDP.params.directed_exp =  0; 
                    MDP.params.random_exp = 1;
                end
            elseif ismember(model, {'KF_SIGMA_DDM', 'KF_SIGMA', 'KF_SIGMA_RACING'})
                    MDP.params.cong_base_info_bonus = 0;
                    MDP.params.incong_base_info_bonus = 0;
                    MDP.params.cong_directed_exp = 0;
                    MDP.params.incong_directed_exp = 0;
                    MDP.params.random_exp = 5;
            end
        end

        % Parameters fixed or fit in Kalman Filter (KF) models
        if ismember(model, {'KF_SIGMA_logistic', 'KF_SIGMA_logistic_DDM','KF_UCB', 'KF_UCB_DDM', 'KF_SIGMA_DDM', 'KF_SIGMA', 'KF_SIGMA_logistic_RACING','KF_SIGMA_RACING'})
            if any(strcmp('sigma_d', MDP.field))
                MDP.params.sigma_d = 6;
            else
                MDP.params.sigma_d = 0;
            end
            MDP.params.sigma_r = 8;
            MDP.params.initial_sigma = 10000;

        % Parameters fixed or fit in RL models    
        elseif ismember(model, {'RL'})
            MDP.params.noise_learning_rate = .1;
            if any(strcmp('associability_weight', MDP.field))
                % associability model
                MDP.params.associability_weight = .1;
                MDP.params.initial_associability = 1;
                MDP.params.learning_rate = .5;
            elseif any(strcmp('learning_rate_pos', MDP.field)) && any(strcmp('learning_rate_neg', MDP.field))
                % split learning rate, no associability
                MDP.params.learning_rate_pos = .5;
                MDP.params.learning_rate_neg = .5;
            else
                % basic RL model, no associability
                MDP.params.learning_rate = .5;
            end
        end
        
        
        % Parameters fixed or fit in DDM or RACING Model
        if contains(model, 'DDM') || contains(model, 'RACING')
            MDP.params.decision_thresh_baseline = 2; 
            if ismember(model, {'KF_SIGMA_logistic_DDM', 'KF_SIGMA_logistic_RACING','KF_SIGMA_RACING', 'obs_means_logistic_DDM'})
                MDP.params.starting_bias_baseline = .5;
                MDP.params.drift_baseline = 0;
                if any(contains(MDP.settings.drift_mapping,'reward_diff'))
                    MDP.params.drift_reward_diff_mod =  .1;
                end
                if any(contains(MDP.settings.bias_mapping,'reward_diff'))
                    MDP.params.starting_bias_reward_diff_mod = .1;
                end
                % Racing Accumulator Params 
                if ismember(model, {'KF_SIGMA_logistic_RACING','KF_SIGMA_RACING'})
                    MDP.params.wd = 0.05;
                    MDP.params.ws = 0.05;
                    MDP.params.V0 = 0;
                end
            elseif ismember(model, {'KF_SIGMA_DDM'})
                    MDP.params.rdiff_bias_mod = .05;
                    MDP.params.decision_thresh_baseline = 3;
            end
        end
       
        % Parameters fixed or fit in kalman filter (KF) logistic model
        if ismember(model, {'KF_SIGMA_logistic','KF_SIGMA_logistic_DDM', 'KF_SIGMA_logistic_RACING', 'obs_means_logistic', 'obs_means_logistic_DDM'})
            MDP.params.info_bonus_small_hor = 0;
            MDP.params.info_bonus_big_hor = 0;
            MDP.params.dec_noise_small_hor = 1;
            MDP.params.dec_noise_big_hor = 1;
            MDP.params.side_bias_small_hor = 0;
            MDP.params.side_bias_big_hor = 0;
            
        end
        % display the MDP.params
        disp(MDP.params)
        
        
        if SIM
            cb = 2; % choose if simulating data using CB1 or CB2 schedule
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
                    plot_choice_given_gen_mean(root, fitting_procedure, experiment,room, results_dir, MDP, id, gen_mean_difference, horizon, truncate_big_hor);
                end
                if do_plot_model_statistics
                    main_plot_model_stats_or_sweep(root, fitting_procedure, experiment, room, results_dir,MDP, id);
                end
            end
            % Indicate if you would like to do model-free analyses on
            % simulated data
            if MDP.do_simulated_model_free
                output_table = get_simulated_model_free(root, fitting_procedure, experiment, room, cb, results_dir,MDP,id);
            end
        end
    end
    if FIT
        output_table = get_fits(root, fitting_procedure, experiment, room, results_dir,MDP, id);
    end
end
