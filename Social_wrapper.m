%% Clear workspace
clear all
SIM = 1;
FIT = 0;
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
    id = '60caf58c38ce3e0f5a51f62b'; % 666878a27888fdd27f529c64 60caf58c38ce3e0f5a51f62b 668d6d380fb72b01a09dee54
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
MDP.params.info_bonus = 5; MDP.params.info_bonus = 0;
MDP.params.random_exp = 5;
MDP.params.side_bias = 0; % unbounded
MDP.params.initial_sigma = 1000;
MDP.params.sigma_r = 4;
MDP.params.familiarity_bonus = 0;
MDP.params.initial_mu = 50;
MDP.params.baseline_noise = 1;

MDP.field = {'sigma_d', 'info_bonus', 'baseline_noise', 'random_exp', 'side_bias', 'sigma_r', 'familiarity_bonus'};

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
    fits_table = get_fits(root, experiment, model, room, results_dir,MDP, id);
end


% Effect of DE - people more likely to pick high info in H6 vs H1
% Effect of RE - people behave more randomly in H5 vs H1. Easier to see when set info_bonus to 0 and gen_mean>4. 
% Increasing confidence within game - can see in some games e.g., game 8
% (gen_mean=4), game 2 (gen_mean=8)