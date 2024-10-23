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
    root = 'L:/';
    run=1;
    experiment = 'prolific'; % indicate local or prolific
    room = 'Like';
    model = 'kf';
    results_dir = sprintf([root 'rsmith/lab-members/cgoldman/Wellbeing/social_media/output/%s/%s/'], experiment, model);
    id = '60caf58c38ce3e0f5a51f62b'; % 666878a27888fdd27f529c64 60caf58c38ce3e0f5a51f62b
elseif ismac
    root = '/Volumes/labs/';
    run=1;
elseif isunix 
    root = '/media/labs/';
    results_dir = getenv('RESULTS');   % run = 1,2,3
    room = getenv('ROOM'); %Like and/or Dislike
    model = getenv('MODEL'); %Like and/or Dislike
    experiment = getenv('EXPERIMENT');
end

addpath([root '/rsmith/all-studies/util/spm12/']);
addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);

%% Set parameters or run loop over all-----
% study = 'prolific'; %prolific or local

MDP.params.sigma_d = .25;
MDP.params.info_bonus = 5; % unbounded
MDP.params.outcome_informativeness = .01;
MDP.params.random_exp = 5;
MDP.params.side_bias = 0; % unbounded
MDP.params.initial_sigma = 100;
MDP.params.initial_sigma_r = 4;
MDP.params.familiarity_bonus = 0;
MDP.params.initial_mu = 50;

MDP.field = {'sigma_d', 'info_bonus', 'outcome_informativeness', 'random_exp', 'side_bias', 'initial_sigma', 'initial_sigma_r', 'familiarity_bonus'};

if SIM
    % choose generative mean difference of 2, 4, 8, 12, 24
    gen_mean_difference = 8; %
    % choose horizon of 1 or 5
    horizon = 5;
    truncate_h5 = 0;
    simulate_social_media(MDP.params, gen_mean_difference, horizon, truncate_h5);
end
if FIT
    fits_table = get_fits(root, experiment, model, room, results_dir,MDP, id);
end
% todo make sure horizon has effect on DE and RE for H5 vs H1
% make sure you see appropriate gradients of increasing confidence within a
% game
% make sure RE has more of an effect on games with low gen mean difference

