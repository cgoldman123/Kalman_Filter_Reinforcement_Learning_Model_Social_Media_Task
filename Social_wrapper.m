function [fits_table] = Social_wrapper(varargin)

%% Clear workspace
clearvars -except varargin
SIM = 0; % Simulate the model
FIT = 1; % Fit the model
MDP.get_rts_and_dont_fit_model = 0; % Get the rts and dont fit the model

rng(23);

% warning('off', 'all');
%% Construct the appropriate path depending on the system this is run on
% If running on the analysis cluster, some parameters will be supplied by 
% the job submission script -- read those accordingly.

dbstop if error
if ispc
    model = "KF_UCB_DDM"; % indicate if RL, KF_UCB, or KF_UCB_DDM
    root = 'L:/';
    experiment = 'prolific'; % indicate local or prolific
    results_dir = sprintf([root 'rsmith/lab-members/cgoldman/Wellbeing/social_media/output/test/']);
    if nargin > 0
        id = varargin{1};
        room = varargin{2};
    else
        id = '5f16f559325a640008bb9a07'; % 666878a27888fdd27f529c64 60caf58c38ce3e0f5a51f62b 668d6d380fb72b01a09dee54 659ab1b4640b25ce093058a2 5590a34cfdf99b729d4f69dc 53b98f20fdf99b472f4700e4
        room = 'Dislike';
    end

    
    MDP.field = {'decision_thresh_baseline','drift_reward_diff_mod','decision_thresh_decision_noise_mod','sigma_d','baseline_noise','side_bias','sigma_r','starting_bias_baseline','drift_baseline','info_bonus','random_exp','baseline_info_bonus'};
    if model == "KF_UCB_DDM"
        % possible mappings are action_prob, reward_diff, UCB,
        % side_bias, decsision_noise
        MDP.settings.drift_mapping = {'reward_diff,side_bias'};
        MDP.settings.thresh_mapping = {'decision_noise'};
        MDP.settings.bias_mapping = {};
        MDP.settings.max_rt = 7;
    end
    
elseif isunix
    model = getenv('MODEL')
    root = '/media/labs/'
    results_dir = getenv('RESULTS')   % run = 1,2,3
    room = getenv('ROOM') %Like and/or Dislike
    experiment = getenv('EXPERIMENT')
    id = getenv('ID')
    MDP.field = strsplit(getenv('FIELD'), ',');
    if model == "KF_UCB_DDM"
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
    end
end

addpath([root '/rsmith/all-studies/util/spm12/']);
addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);


model_functions = containers.Map(...
    {'KF_UCB', 'RL', 'KF_UCB_DDM'}, ...
    {@model_SM_KF_all_choices, @model_SM_RL_all_choices, @model_SM_KF_DDM_all_choices} ...
);
if isKey(model_functions, model)
    MDP.model = model_functions(model);
else
    error('Unknown model specified in MDP.model');
end




% parameters fit across models
MDP.params.side_bias = 0; 
MDP.params.baseline_info_bonus = 0; 
MDP.params.baseline_noise = .1;
MDP.params.reward_sensitivity = 1;
MDP.params.initial_mu = 50;

% make directed exploration and random exploration same param or keep
% together
if any(strcmp('DE_RE_horizon', MDP.field))
    MDP.params.DE_RE_horizon = 2.5; % prior on this value
else
    if any(strcmp('info_bonus', MDP.field))
        MDP.params.info_bonus = 5; 
    else
        MDP.params.info_bonus = 0; 
    end
    if any(strcmp('random_exp', MDP.field))
        MDP.params.random_exp = 2.5;
    else
        MDP.params.random_exp = 0;
    end
end


% parameters specific to one of the models
if ismember(model, {'KF_UCB', 'KF_UCB_DDM'})
    MDP.params.sigma_d = .25;
    MDP.params.initial_sigma = 1000;
    MDP.params.sigma_r = 4;
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

if ismember(model, {'KF_UCB_DDM'})
    % set drift params
    MDP.params.drift_baseline = 0;
    if any(contains(MDP.settings.drift_mapping,'action_prob'))
        MDP.params.drift_action_prob_mod = .1;  
    end
    if any(contains(MDP.settings.drift_mapping,'reward_diff'))
        MDP.params.drift_reward_diff_mod = .1;
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

% display the MDP.params
disp(MDP.params)


if SIM
    % choose generative mean difference of 2, 4, 8, 12, 24
    gen_mean_difference = 8; %
    % choose horizon of 1 or 5
    horizon = 5;
    % if truncate_h5 is true, use the H5 bandit schedule but truncate so that all games are H1
    truncate_h5 = 0;
    simulate_social_media(MDP.params, gen_mean_difference, horizon, truncate_h5);
end
if FIT
    fits_table = get_fits(root, experiment, room, results_dir,MDP, id);
end
end

% Effect of DE - people more likely to pick high info in H5 vs H1
% Effect of RE - people behave more randomly in H5 vs H1. Easier to see when set info_bonus to 0 and gen_mean>4. 
% Increasing confidence within game - can see in some games e.g., game 8
% (gen_mean=4), game 2 (gen_mean=8)