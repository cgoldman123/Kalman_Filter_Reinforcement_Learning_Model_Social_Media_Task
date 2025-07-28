function [output_table] = Social_wrapper(varargin)
    %% If running the pyddm scripts for the first time, make sure the correct python virtual environment is called. Run this code to activate the correct virtual environment
    % pyenv('Version', 'C:\Users\CGoldman\AppData\Local\anaconda3\envs\pyddm\python.exe')
    
    %% Clear workspace
    clearvars -except varargin
    SIM = 0; % Simulate the model
    FIT = 1; % Fit the model
    if FIT
        MDP.get_rts_and_dont_fit_model = 0; % Toggle on to extract the rts and not fit the model
        MDP.do_model_free = 1; % Toggle on to do model-free analyses on actual data
        MDP.fit_model = 1; % Toggle on to fit the model
        if MDP.fit_model
            MDP.do_simulated_model_free = 1; % Toggle on to do model-free analyses on data simulated from posterior parameter estimates of model.
        end
    elseif SIM
        MDP.plot_simulated_data = 1; %Toggle on to plot data simulated by model
        MDP.do_simulated_model_free = 1; % Toggle on to do model-free analyses on data simulated from prior parameters initialized in this main file.
        id_label = '562c2ff0733ea000111630df_Iteration_5'; % Use this to give a name to the simulated data
    end
    rng(23);
    
    % warning('off', 'all');
    %% Construct the appropriate path depending on the system this is run on
    % If running on the analysis cluster, some parameters will be supplied by 
    % the job submission script -- read those accordingly.
    
    dbstop if error
    if ispc
        fitting_procedure = "SPM"; % Specify fitting procedure as "SPM", "VBA", or "PYDDM"
        root = 'L:/';
        experiment = 'local'; % indicate local or prolific
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
            model = "KF_SIGMA_logistic_RACING"; % indicate if 'KF_UCB', 'RL', 'KF_UCB_DDM', 'KF_SIGMA_DDM', 'KF_SIGMA'
            MDP.field = {'h1_dec_noise','h5_dec_noise','side_bias_h1','side_bias_h5','h5_info_bonus','h1_info_bonus','sigma_d', 'sigma_r', 'starting_bias_baseline', 'drift_baseline', 'decision_thresh_baseline', 'drift_reward_diff_mod', 'wd', 'ws', 'V0'};
            if strcmp(fitting_procedure, "VBA")
                MDP.observation_params = MDP.field; % When there is no latent state learning, all params are observation params
            end
            if ismember(model, {'KF_UCB_DDM', 'KF_SIGMA_DDM', 'KF_SIGMA_logistic_DDM', 'KF_SIGMA_logistic_RACING'})
                % possible mappings are action_prob, reward_diff, UCB,
                % side_bias, decsision_noise
                MDP.settings.drift_mapping = {'reward_diff','decision_noise'};
                MDP.settings.bias_mapping = {'info_diff,side_bias'};
                MDP.settings.thresh_mapping = {''};
                MDP.settings.max_rt = 7;
            else
                MDP.settings = '';
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
    
            if ismember(model, {'KF_UCB_DDM', 'KF_SIGMA_DDM'})
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
    addpath(['./SPM_scripts/'])
    addpath(['./VBA_scripts/'])
    addpath(['./racing_accumulator/'])
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
            {'KF_SIGMA_logistic','KF_SIGMA_logistic_DDM','KF_UCB', 'RL', 'KF_UCB_DDM', 'KF_SIGMA_DDM', 'KF_SIGMA' 'KF_SIGMA_logistic_RACING'}, ...
            {@model_SM_KF_SIGMA_logistic, @model_SM_KF_SIGMA_logistic_DDM,@model_SM_KF_all_choices, @model_SM_RL_all_choices, @model_SM_KF_DDM_all_choices, @model_SM_KF_SIGMA_DDM_all_choices, @model_SM_KF_SIGMA_all_choices @model_SM_KF_SIGMA_logistic_RACING} ...
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
        if ismember(model, {'KF_UCB', 'KF_UCB_DDM', 'KF_SIGMA_DDM', 'KF_SIGMA', 'RL'})
            MDP.params.side_bias =  0; 
            MDP.params.baseline_info_bonus =  0; 
            MDP.params.baseline_noise = 1/12; 
            
            % make directed exploration and random exploration same param or keep
            % together
            if any(strcmp('DE_RE_horizon', MDP.field))
                MDP.params.DE_RE_horizon = 2.5; % prior on this value
            else
                MDP.params.directed_exp =  0; 
                MDP.params.random_exp = 1;
            end
        end

        % Parameters fixed or fit in Kalman Filter (KF) models
        if ismember(model, {'KF_SIGMA_logistic', 'KF_SIGMA_logistic_DDM','KF_UCB', 'KF_UCB_DDM', 'KF_SIGMA_DDM', 'KF_SIGMA', 'KF_SIGMA_logistic_RACING'})
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
        
        % Parameters fixed or fit in Kalman Filter (KF) UCB DDM Model
        if ismember(model, {'KF_UCB_DDM'})
            % set drift params
            MDP.params.drift_baseline = 0;
            if any(contains(MDP.settings.drift_mapping,'action_prob'))
                MDP.params.drift_action_prob_mod = .1;  
            end
            if any(contains(MDP.settings.drift_mapping,'reward_diff'))
                MDP.params.drift_reward_diff_mod =.1;
            end
            if any(contains(MDP.settings.drift_mapping,'UCB_diff'))
                MDP.params.drift_UCB_diff_mod = .1;
            end
            
            % set starting bias params
            MDP.params.starting_bias_baseline = .5;
            if any(contains(MDP.settings.bias_mapping,'action_prob'))
                MDP.params.starting_bias_action_prob_mod = .1;  
            end
            if any(contains(MDP.settings.bias_mapping,'reward_diff'))
                MDP.params.starting_bias_reward_diff_mod = .1;
            end
            if any(contains(MDP.settings.bias_mapping,'UCB_diff'))
                MDP.params.starting_bias_UCB_diff_mod = .1;
            end
        
            % set decision threshold params
            MDP.params.decision_thresh_baseline = .5;
            if any(contains(MDP.settings.thresh_mapping,'action_prob'))
                MDP.params.decision_thresh_action_prob_mod = .1;  
            end
            if any(contains(MDP.settings.thresh_mapping,'reward_diff'))
                MDP.params.decision_thresh_reward_diff_mod = .1;
            end
            if any(contains(MDP.settings.thresh_mapping,'UCB_diff'))
                MDP.params.decision_thresh_UCB_diff_mod = .1;
            end    
            if any(contains(MDP.settings.thresh_mapping,'decision_noise'))
                MDP.params.decision_thresh_decision_noise_mod = 1;
            end
        end
        
        % Parameters fixed or fit in Kalman Filter (KF) SIGMA DDM Model
        if ismember(model, {'KF_SIGMA_DDM','KF_SIGMA_logistic_DDM', 'KF_SIGMA_logistic_RACING'})
            MDP.params.starting_bias_baseline = .5;
            MDP.params.drift_baseline = 0;
            MDP.params.decision_thresh_baseline = 2; 
            if any(contains(MDP.settings.drift_mapping,'reward_diff'))
                MDP.params.drift_reward_diff_mod =  .1;
            end
            if any(contains(MDP.settings.bias_mapping,'reward_diff'))
                MDP.params.starting_bias_reward_diff_mod = .1;
            end
            %% Racing Accumulator Params %%
            if ismember(model, {'KF_SIGMA_logistic_RACING'})
                MDP.params.wd = 0.05;
                MDP.params.ws = 0.05;
                MDP.params.V0 = 0;
            end
        end
       
        % Parameters fixed or fit in Kalman Filter (KF) logistic model
        if ismember(model, {'KF_SIGMA_logistic','KF_SIGMA_logistic_DDM', 'KF_SIGMA_logistic_RACING'})
            MDP.params.h1_info_bonus = 0;
            MDP.params.h5_info_bonus = 0;
            MDP.params.h1_dec_noise = 1;
            MDP.params.h5_dec_noise = 1;
            MDP.params.side_bias_h1 = 0;
            MDP.params.side_bias_h5 = 0;
            
        end
        % display the MDP.params
        disp(MDP.params)
        
        
        if SIM
            if MDP.plot_simulated_data
                % choose generative mean difference of 2, 4, 8, 12, 24
                gen_mean_difference = 4; %
                % choose horizon of 1 or 5
                horizon = 5;
                % if truncate_h5 is true, use the H5 bandit schedule but truncate so that all games are H1
                truncate_h5 = 1;
                plot_simulated_behavior(MDP, gen_mean_difference, horizon, truncate_h5);
            end
            if MDP.do_simulated_model_free
                cb = 2; % choose if simulating CB1 or CB2
                output_table = get_simulated_model_free(root, experiment, room, cb, results_dir,MDP,id_label);
            end
        end
    end
    if FIT
        output_table = get_fits(root, fitting_procedure, experiment, room, results_dir,MDP, id);
    end
end


% Effect of DE - people more likely to pick high info in H5 vs H1
% Effect of RE - people behave more randomly in H5 vs H1. Easier to see when set info_bonus to 0 and gen_mean>4. 
% Increasing confidence within game - can see in some games e.g., game 8
% (gen_mean=4), game 2 (gen_mean=8)