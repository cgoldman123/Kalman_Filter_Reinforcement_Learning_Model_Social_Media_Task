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
    id = '666878a27888fdd27f529c64'; % 666878a27888fdd27f529c64 60caf58c38ce3e0f5a51f62b
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

MDP.params.alpha_start = .5; % bound between 0 and 1
MDP.params.alpha_inf = .5; % bound between 0 and 1
MDP.params.info_bonus_h1 = 0; % unbounded
MDP.params.info_bonus_h5 = 0; % unbounded
MDP.params.dec_noise_h1_13 = 1; % bound positive
MDP.params.dec_noise_h5_13 = 1; % bound positive
MDP.params.side_bias_h1 = 0; % unbounded
MDP.params.side_bias_h5 = 0; % unbounded
MDP.field = fieldnames(MDP.params);


get_fits(root, experiment, model, room, results_dir,MDP, id);




