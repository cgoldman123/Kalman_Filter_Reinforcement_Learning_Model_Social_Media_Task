%% Clear workspace
clear all
SIM = 0;
FIT = 1;
rng(23);
%% Construct the appropriate path depending on the system this is run on
% If running on the analysis cluster, some parameters will be supplied by 
% the job submission script -- read those accordingly.

dbstop if error
if ispc
    model = "KF_UCB_DDM"; % indicate if RL, KF_UCB, or KF_UCB_DDM
    root = 'L:/';
    experiment = 'prolific'; % indicate local or prolific
    room = 'Like';
    results_dir = sprintf([root 'rsmith/lab-members/cgoldman/Wellbeing/social_media/output/test/']);
    id = '60caf58c38ce3e0f5a51f62b'; % 666878a27888fdd27f529c64 60caf58c38ce3e0f5a51f62b 668d6d380fb72b01a09dee54 659ab1b4640b25ce093058a2 5590a34cfdf99b729d4f69dc 53b98f20fdf99b472f4700e4
    
    MDP.field = {'sigma_r', 'side_bias', 'decision_thresh_baseline', 'starting_bias', 'drift_baseline', 'drift_action_prob_mod', 'h1_info_bonus', 'h5_slope_info_bonus', 'h5_baseline_info_bonus', 'h1_dec_noise', 'h5_slope_dec_noise', 'h5_baseline_dec_noise'};
    if model == "KF_UCB_DDM"
        % possible mappings are action_prob, reward_diff, UCB,
        % side_bias, decsision_noise
        MDP.settings.drift_mapping = {'action_prob'};
        MDP.settings.thresh_mapping = {};
        MDP.settings.bias_mapping = {};
        MDP.settings.max_rt = 3;
    end
    
elseif isunix
    model = getenv('MODEL')
    root = '/media/labs/'
    results_dir = getenv('RESULTS')   % run = 1,2,3
    room = getenv('ROOM') %Like and/or Dislike
    experiment = getenv('EXPERIMENT')
    id = getenv('ID')
    MDP.field = strsplit(getenv('FIELD'), ',')
    if model == "KF_UCB_DDM"
        MDP.settings.drift_mapping = strsplit(getenv('DRIFT_MAPPING'), ','); display(MDP.settings.drift_mapping)
        MDP.settings.bias_mapping = strsplit(getenv('BIAS_MAPPING'), ','); display(MDP.settings.bias_mapping)
        MDP.settings.thresh_mapping = strsplit(getenv('THRESH_MAPPING'), ','); display(MDP.settings.thresh_mapping)
        MDP.settings.max_rt = 3;

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
MDP.params.h1_info_bonus = 0; 
MDP.params.h1_dec_noise = 1;
MDP.params.h5_slope_dec_noise = 0;
MDP.params.h5_slope_info_bonus = 0;

MDP.params.reward_sensitivity = 1;
MDP.params.initial_mu = 50;
if any(strcmp('DE_RE_horizon', MDP.field))
    MDP.params.DE_RE_horizon = 2.5; % prior on this value
else
    if any(strcmp('h5_baseline_info_bonus', MDP.field))
        MDP.params.h5_baseline_info_bonus = 0; 
    else
        MDP.params.h5_baseline_info_bonus = 0; 
    end
    if any(strcmp('h5_baseline_dec_noise', MDP.field))
        MDP.params.h5_baseline_dec_noise = 1;
    else
        MDP.params.h5_baseline_dec_noise = 0;
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
    if isempty(MDP.settings.drift_mapping)
        MDP.params.drift_baseline = 0;
    else
        MDP.params.drift_baseline = .085;
        if contains(MDP.settings.drift_mapping,'action_prob')
            MDP.params.drift_action_prob_mod = .5;  
        end
        if contains(MDP.settings.drift_mapping,'reward_diff')
            MDP.params.drift_reward_diff_mod = .5;
        end
        if contains(MDP.settings.drift_mapping,'UCB_diff')
            MDP.params.drift_UCB_diff_mod = .5;
        end
    end

    if isempty(MDP.settings.bias_mapping)
        MDP.params.starting_bias_baseline = .5;
    else
        MDP.params.starting_bias_baseline = .5;
        if contains(MDP.settings.bias_mapping,'action_prob')
            MDP.params.bias_action_prob_mod = .5;  
        end
        if contains(MDP.settings.bias_mapping,'reward_diff')
            MDP.params.bias_reward_diff_mod = .5;
        end
        if contains(MDP.settings.bias_mapping,'UCB_diff')
            MDP.params.bias_UCB_diff_mod = .5;
        end
    end

    if isempty(MDP.settings.thresh_mapping)
        MDP.params.decision_thresh_baseline = 2;
    else
        MDP.params.decision_thresh_baseline = .085;
        if contains(MDP.settings.thresh_mapping,'action_prob')
            MDP.params.thresh_action_prob_mod = .5;  
        end
        if contains(MDP.settings.thresh_mapping,'reward_diff')
            MDP.params.thresh_reward_diff_mod = .5;
        end
        if contains(MDP.settings.thresh_mapping,'UCB_diff')
            MDP.params.thresh_UCB_diff_mod = .5;
        end
    end
end

if ismember(model, {'RL'})
    MDP.params.noise_learning_rate = .1;
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


% Effect of DE - people more likely to pick high info in H5 vs H1
% Effect of RE - people behave more randomly in H5 vs H1. Easier to see when set info_bonus to 0 and gen_mean>4. 
% Increasing confidence within game - can see in some games e.g., game 8
% (gen_mean=4), game 2 (gen_mean=8)