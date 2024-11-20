%% Clear workspace
clear all
SIM = 0;
FIT = 1;
rng(23);
%% Construct the appropriate path depending on the system this is run on
% If running on the analysis cluster, some parameters will be supplied by 
% the job submission script -- read those accordingly.

dbstop if error

model = "RL";
model_functions = containers.Map(...
    {'KF', 'RL'}, ...
    {@model_SM_KF_all_choices, @model_SM_RL_all_choices} ...
);
% Save the selected function handle to a variable
if isKey(model_functions, model)
    MDP.model = model_functions(model);
else
    error('Unknown model specified in MDP.model');
end


if ispc
    root = 'L:/';
    experiment = 'prolific'; % indicate local or prolific
    room = 'Like';
    model = 'UCB';
    results_dir = sprintf([root 'rsmith/lab-members/cgoldman/Wellbeing/social_media/output/%s/%s/'], experiment, model);
    id = '659ab1b4640b25ce093058a2'; % 666878a27888fdd27f529c64 60caf58c38ce3e0f5a51f62b 668d6d380fb72b01a09dee54 659ab1b4640b25ce093058a2
    MDP.field = {'learning_rate', 'baseline_noise', 'side_bias', 'baseline_info_bonus', 'info_bonus', 'random_exp','associability_weight', 'initial_associability' };
    MDP.field = {'learning_rate_pos','learning_rate_neg', 'baseline_noise', 'side_bias', 'baseline_info_bonus', 'info_bonus', 'random_exp' };
    MDP.field = {'learning_rate', 'baseline_noise', 'side_bias', 'baseline_info_bonus', 'info_bonus', 'random_exp' };
    MDP.field = {'learning_rate', 'baseline_noise', 'side_bias', 'baseline_info_bonus', 'DE_RE_horizon' };

elseif ismac
    root = '/Volumes/labs/';
elseif isunix 
    root = '/media/labs/'
    results_dir = getenv('RESULTS')   % run = 1,2,3
    room = getenv('ROOM') %Like and/or Dislike
    experiment = getenv('EXPERIMENT')
    id = getenv('ID')
    MDP.field = strsplit(getenv('FIELD'), ',')

end

addpath([root '/rsmith/all-studies/util/spm12/']);
addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);

%% Set parameters or run loop over all-----

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

% learning
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


MDP.params.initial_mu = 50;
MDP.params.reward_sensitvity = 1;

% choice
MDP.params.side_bias = 0; 
MDP.params.baseline_info_bonus = 0; 
MDP.params.baseline_noise = 1;
MDP.params.noise_learning_rate = .1;


if SIM
    % choose generative mean difference of 2, 4, 8, 12, 24
    gen_mean_difference = 8; %
    % choose horizon of 1 or 5
    horizon = 5;
    % if truncate_h5 is true, use the H5 bandit schedule but truncate so that
    %all games are H1
    truncate_h5 = 0;
    simulate_social_media(MDP.params, gen_mean_difference, horizon, truncate_h5);
end
if FIT
    fits_table = get_fits(root, experiment, room, results_dir,MDP, id);
end


% Effect of DE - people more likely to pick high info in H6 vs H1
% Effect of RE - people behave more randomly in H5 vs H1. Easier to see when set info_bonus to 0 and gen_mean>4. 
% Increasing confidence within game - can see in some games e.g., game 8
% (gen_mean=4), game 2 (gen_mean=8)