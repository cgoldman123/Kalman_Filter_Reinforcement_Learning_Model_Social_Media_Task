function [fits, model_output] = fit_extended_model(formatted_file, result_dir, MDP)
    if ispc
        root = 'L:/';
    else
        root = '/media/labs/';
    end
    fprintf('Using this formatted_file: %s\n',formatted_file);

    %formatted_file = 'L:\rsmith\wellbeing\tasks\SocialMedia\output\prolific\kf\beh_Dislike_06_03_24_T16-03-52.csv';  %% remember to comment out
    addpath([root 'rsmith/lab-members/cgoldman/general/']);
    %addpath('~/Documents/MATLAB/MatJAGS/');


    
    fundir      = pwd;%[maindir 'TMS_code/'];
    datadir     = pwd;%[maindir 'TMS_code/'];
    savedir     = pwd;%[maindir];
    addpath(fundir);
    cd(fundir);
    %defaultPlotParameters

%     sub = load_TMS_v1([datadir '/EIT_HorizonTaskOutput_HierarchicalModelFormat_v2.csv']);
    sub = load_TMS_v1(formatted_file);
%     sub = sub(1,:); % REMEMBER TO COMMENT OUT
    disp(sub);
    
    %% ========================================================================
    %% HIERARCHICAL MODEL FIT ON FIRST FREE CHOICE %%
    %% HIERARCHICAL MODEL FIT ON FIRST FREE CHOICE %%
    %% HIERARCHICAL MODEL FIT ON FIRST FREE CHOICE %%
    %% ========================================================================

    %% prep data structure 
    clear a
    L = unique(sub(1).gameLength);
    i = 1;
    NS = length(sub);   % number of subjects
    T = 4;              % number of forced choices
    U = 1;  %used to be 2            % number of uncertainty conditions
    
    NUM_GAMES = 40; %max(vertcat(sub.game), [], 'all');
    
    a  = zeros(NS, NUM_GAMES, T);
    c5 = nan(NS,   NUM_GAMES);
    r  = zeros(NS, NUM_GAMES, T+1); % CMG changed last dimension to T+1 to be able to get prediction error for free choice
    GL = nan(NS,   NUM_GAMES);

    for sn = 1:length(sub)

        % choices on forced trials
        dum = sub(sn).a(:,1:4);
        a(sn,1:size(dum,1),:) = dum;

        % choices on free trial
        % note a slight hacky feel here - a is 1 or 2, c5 is 0 or 1.
        dum = sub(sn).a(:,5) == 2;
        L(sn) = length(dum);
        c5(sn,1:size(dum,1)) = dum;

        % rewards
        dum = sub(sn).r(:,1:5); % CMG changed last dimension to T+1 to be able to get prediction error for free choice
        r(sn,1:size(dum,1),:) = dum;

        % game length
        dum = sub(sn).gameLength;
        GL(sn,1:size(dum,1)) = dum;

        G(sn) = length(dum);

        % uncertainty condition 
        dum = abs(sub(sn).uc - 2) + 1;
        UC(sn, 1:size(dum,1)) = dum;

        % difference in information
        dum = sub(sn).uc - 2;
        dI(sn, 1:size(dum,1)) = -dum;

        % TMS flag
        dum = strcmp(sub(sn).expt_name, 'RFPC');
        TMS(sn,1:size(dum,1)) = dum;
        

    end

    dum = GL(:); dum(dum==0) = [];
    H = length(unique(dum));
    dum = UC(:); dum(dum==0) = [];
    U = length(unique(dum));
    GL(GL==5) = 1;
    GL(GL==9) = 2; %used to be 10

    C1 = GL ;      %(GL-1)*2+UC;      CAL edits
    C2 = TMS + 1;
    nC1 = 2;
    nC2 = 1;

    % meaning of condition 1
    % gl uc c1
    %  1  1  1 - horizon 1, [2 2]
    %  1  2  2 - horizon 6, [1 3]
    %  2  1  3 - horizon 1, [2 2]
    %  2  2  4 - horizon 6, [1 3]

    % meaning of condition 1 (SMT FIXED)
    % gl uc c1
    %  1  1  1 - horizon 1, [2 2]
    %  1  2  2 - horizon 1, [1 3]
    %  2  1  3 - horizon 6, [2 2]
    %  2  2  4 - horizon 6, [1 3]



    datastruct = struct(...
        'C1', C1, 'nC1', nC1, ...
        'NS', NS, 'G',  G,  'T',   T, ...
        'dI', dI, 'a',  a,  'c5',  c5, 'r', r, 'result_dir', result_dir);
    

    if ispc
        root = 'L:/';
    elseif ismac
        root = '/Volumes/labs/';
    elseif isunix 
        root = '/media/labs/';
    end
    
    fprintf( 'Running Newton Function to fit\n' );
    MDP.datastruct = datastruct;

    DCM = SM_inversion(MDP);
   

    
    %% Organize fits
    field = DCM.field;
    % get fitted and fixed params
    fits = DCM.params;
    for i = 1:length(field)
        if ismember(field{i},{'alpha_start', 'alpha_inf'})
            fits.(field{i}) = 1/(1+exp(-DCM.Ep.(field{i})));
        elseif ismember(field{i},{'dec_noise_h1_13', 'dec_noise_h5_13'})
            fits.(field{i}) = exp(DCM.Ep.(field{i}));
        elseif ismember(field{i},{'info_bonus_h1', 'info_bonus_h5','side_bias_h1', 'side_bias_h5'})
            fits.(field{i}) = DCM.Ep.(field{i});
        else
            disp(field{i});
            error("Param not propertly transformed");
        end
    end
    
    
    
    free_choices = DCM.datastruct.c5(1,:);
    rewards = squeeze(DCM.datastruct.r(1,:,:))';

    mdp.horizon_sequence = DCM.datastruct.C1(1,:);
    mdp.forced_choices = squeeze(DCM.datastruct.a(1,:,:))';
    mdp.right_info = squeeze(DCM.datastruct.dI(1,:,:));
    mdp.T = 4; % num forced choices
    mdp.G = 40; % game length

    % note that mu2 == right bandit ==  c=2 == free choice = 1


    %model_output(si).results = model_KFcond_v2_SMT_CMG(params,free_choices, rewards,mdp);    
    model_output = model_KFcond_v3_CMG(fits,free_choices, rewards,mdp);    
    fits.average_action_prob = mean(model_output.action_probs);
    fits.model_acc = sum(model_output.action_probs > .5) / mdp.G;
        
    
    
    
    datastruct = struct(...
        'C1', C1, 'nC1', nC1, ...
        'NS', NS, 'G',  G,  'T',   T, ...
        'dI', dI, 'a',  a,  'c5',  model_output.simmed_free_choices, 'r', r, 'result_dir', result_dir);
    
    MDP.datastruct = datastruct;
    % note old social media model model_KFcond_v2_SMT
    fprintf( 'Running VB to fit simulated behavior! \n' );

    simfit_DCM = SM_inversion(MDP);

    for i = 1:length(field)
        if ismember(field{i},{'alpha_start', 'alpha_inf'})
            fits.(['simfit_' field{i}]) = 1/(1+exp(-simfit_DCM.Ep.(field{i})));
        elseif ismember(field{i},{'dec_noise_h1_13', 'dec_noise_h5_13'})
            fits.(['simfit_' field{i}]) = exp(simfit_DCM.Ep.(field{i}));
        elseif ismember(field{i},{'info_bonus_h1', 'info_bonus_h5','side_bias_h1', 'side_bias_h5'})
            fits.(['simfit_' field{i}]) = simfit_DCM.Ep.(field{i});
        else
            disp(field{i});
            error("Param not propertly transformed");
        end
    end
    
    
 end