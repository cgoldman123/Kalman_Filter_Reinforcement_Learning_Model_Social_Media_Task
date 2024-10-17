%% Clear workspace
clear all

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
% model = 'new'; %old or new

MDP.params.sigma_d = .5;
MDP.params.info_bonus = 5; % unbounded
MDP.params.outcome_informativeness = 1;
MDP.params.random_exp = 5;
MDP.params.side_bias = 0; % unbounded
MDP.params.initial_sigma = 8;
MDP.params.initial_sigma_r = 8;
MDP.params.familiarity_bonus = 0;

MDP.field = fieldnames(MDP.params);


get_fits(root, experiment, model, room, results_dir,MDP, id);




