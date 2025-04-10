function [fit_info, model_output] = fit_extended_model_VBA(formatted_file, result_dir, MDP)
    if ispc
        root = 'L:/';
    else
        root = '/media/labs/';
    end
    fprintf('Using this formatted_file: %s\n',formatted_file);

    %formatted_file = 'L:\rsmith\wellbeing\tasks\SocialMedia\output\prolific\kf\beh_Dislike_06_03_24_T16-03-52.csv';  %% remember to comment out
    addpath([root 'rsmith/lab-members/cgoldman/general/']);

    sub = load_TMS_v1(formatted_file);

    % If we are just getting the rts and not fitting the model, return
    if MDP.get_rts_and_dont_fit_model
        fit_info = sub.RT;
        model_output.results.RT = sub.RT;
        return;
    end
    
    NS = length(sub);   % number of subjects
    T = 4;              % number of forced choices
    
    NUM_GAMES = 40; %max(vertcat(sub.game), [], 'all');
    
    GL = nan(NS,   NUM_GAMES);

    for sn = 1:length(sub)
        % game length
        dum = sub(sn).gameLength;
        GL(sn,1:size(dum,1)) = dum;
        G(sn) = length(dum);
        % information difference
        dum = sub(sn).uc - 2;
        dI(sn, 1:size(dum,1)) = -dum;
    end

    GL(GL==5) = 1;
    GL(GL==9) = 2; %used to be 10
    C1 = GL ;      %(GL-1)*2+UC;      CAL edits
    nC1 = 2;


    %% Prepare data structure for VBA%%

    in = struct(...
        'C1', C1, 'nC1', nC1, ...
        'NS', NS, 'G',  G,  'T',   T, ...
        'dI', dI, 'actions',  sub.a,  'RTs', sub.RT, 'rewards', sub.r, 'bandit1_schedule', sub.bandit1_schedule,...
        'bandit2_schedule', sub.bandit2_schedule, 'MDP', MDP, 'result_dir', result_dir);
    
    fprintf( 'Running VBA to fit\n' );

    % assemble actions variable y
    % 0 indicates the left bandit was chosen, 1 indicates the right
    actions =  [reshape(sub.a.', 1, [])] - 1; % reshape the actions variable to a row vector and shift by 1
    y = actions(~isnan(actions)); % remove NaN values

    % assemble input variable u
    % row 1: actions
    % row 2: rewards
    % row 3: timestep within a game (1 to 9)
    % row 4: horizon (1 or 5)
    input = [actions; reshape(sub.r.', 1, []); repmat(1:9, 1, 40)]; 
    input = input(:, ~isnan(input(1,:))); % remove NaN values
    input = [input; repmat(repelem([1, 5], [5, 9]), 1, 20)];   % create a row to indicate Horizon
    input = [NaN(4,1), input]; % make the first column NaN values because participants choose a bandit before learning
    input(:,find(input(3,:) == 1)-1) = NaN; % remove the result of the last free choice for each game because it's meaningless
    u = input(:,1:end-1);


    % Note that Y must be a sparse double that has as many rows (p) as the number of columns returned
    % by g_fname (p), a function that outputs the probability distribution for
    % that trial.

    f_fname = @f_KF_SIGMA4;
    g_fname = @g_KF_SIGMA4_logistic;

    

    dim = struct( ...
        'n', 4, ... number of hidden states 
        'n_theta', length(MDP.evolution_params), ... number of evolution parameters 
        'n_phi', length(MDP.observation_params) ... number of observation parameters 
       );


    options.priors.muX0 = [50; 50; 10000; 10000]; % prior means for the latent states (initial expected value for each bandit and prior variance)
    in.prior_muX0 = options.priors.muX0;

    [prior_means_evolution_params, prior_sigmas_evolution_params] = transform_params_SM("transform", MDP.params,MDP.evolution_params);
    options.priors.muTheta = spm_vec(prior_means_evolution_params);
    options.priors.SigmaTheta = diag(spm_vec(prior_sigmas_evolution_params)');


    %% TESTING
    %MDP.params.baseline_noise = 10;
    %%%

    [prior_means_observation_params, prior_sigmas_observation_params] = transform_params_SM("transform", MDP.params,MDP.observation_params);
    options.priors.muPhi = spm_vec(prior_means_observation_params);
    options.priors.SigmaPhi = diag(spm_vec(prior_sigmas_observation_params)');


    options.inF = in;
    options.inG = in;
    options.sources.type = 1; % Use source 1 for fitting binary data

    % Normally, the expected first observation is g(x1), ie. after
    % a first iteratition x1 = f(x0, u0). The skipf flag will prevent this evolution
    % and thus set x1 = x0
    options.skipf = repmat([1, zeros(1,4), 1, zeros(1,8)],1,20); 


    % if options.isYout = 1, the datapoint is ignored. Use this to ignore
    % fitting forced choices.
    options.isYout = repmat([ones(1,4), 0, ones(1,4), zeros(1,5)], 1, 20);

    % split into sessions (blocks), parameters and the initial state are carried over 
    options.multisession.split = repmat([5 9],1,20); 
    % By default, all parameters are duplicated for each session. However, you
    % can fix some parameters so they remain constants across sessions.
    % ame evolution parameter in both sessions
    options.multisession.fixed.theta = 'all'; % <~ uncomment for fixing theta params
    
    % Same observation parameter in both sessions
    options.multisession.fixed.phi = 'all'; % <~ uncomment for fixing phi params
    
    % Same initial state in both sessions
    options.multisession.fixed.X0 = 1:dim.n; % <~ this has no effect, but because we are not fitting X0, we will still use the same X0 across sessions
    options.updateX0 = 0 ; % this prevents us from fitting the initial condition


    [posterior, out] = VBA_NLStateSpaceModel(y, u, f_fname, g_fname, dim, options);


    fit_info = MDP.params;

    if ~isempty(MDP.observation_params)
        phi_means_struct = cell2struct(num2cell(posterior.muPhi), MDP.observation_params);
        posterior_means_phi = transform_params_SM("untransform", phi_means_struct,MDP.observation_params); 
        phi_fields = fieldnames(posterior_means_phi);
        for i = 1:length(phi_fields)
            fit_info.(phi_fields{i}) = posterior_means_phi.(phi_fields{i});
        end
    end

      
    if ~isempty(MDP.evolution_params)
        theta_means_struct = cell2struct(num2cell(posterior.muTheta), MDP.evolution_params);
        posterior_means_theta = transform_params_SM("untransform", theta_means_struct,MDP.evolution_params); 
        theta_fields = fieldnames(posterior_means_theta);
        for i = 1:length(theta_fields)
            fit_info.(theta_fields{i}) = posterior_means_theta.(theta_fields{i});
        end
    end
    
    actions_and_rts.actions = in.actions;
    actions_and_rts.RTs = in.RTs;
    rewards = in.rewards;
    
    model_output = MDP.model(fit_info,actions_and_rts, rewards,in, 0);    
    model_output.out = out;

    fit_info.average_action_prob = mean(model_output.action_probs(~isnan(model_output.action_probs)), 'all');
    
    fit_info.average_action_prob_H1_1 = mean(model_output.action_probs(1:2:end, 5), 'omitnan');
    fit_info.average_action_prob_H5_1 = mean(model_output.action_probs(2:2:end, 5), 'omitnan');
    fit_info.average_action_prob_H5_2 = mean(model_output.action_probs(2:2:end, 6), 'omitnan');
    fit_info.average_action_prob_H5_3 = mean(model_output.action_probs(2:2:end, 7), 'omitnan');
    fit_info.average_action_prob_H5_4 = mean(model_output.action_probs(2:2:end, 8), 'omitnan');
    fit_info.average_action_prob_H5_5 = mean(model_output.action_probs(2:2:end, 9), 'omitnan');

    
    fit_info.model_acc = sum(model_output.action_probs(~isnan(model_output.action_probs)) > 0.5) / numel(model_output.action_probs(~isnan(model_output.action_probs)));
    fit_info.F = out.F;
    %fit_info.F = DCM.F;

    if ismember(func2str(MDP.model), {'model_SM_KF_DDM_all_choices', 'model_SM_KF_SIGMA_DDM_all_choices'})
        fit_info.num_invalid_rts = model_output.num_invalid_rts;
    end

    
                
    % simulate behavior with fitted params
    simmed_model_output = MDP.model(fit_info,actions_and_rts, rewards,in, 1);    

    % assemble actions variable y
    % 0 indicates the left bandit was chosen, 1 indicates the right
    actions =  [reshape(simmed_model_output.actions.', 1, [])] - 1; % reshape the actions variable to a row vector and shift by 1
    y = actions(~isnan(actions)); % remove NaN values

    % assemble input variable u
    % row 1: actions
    % row 2: rewards
    % row 3: timestep within a game (1 to 9)
    % row 4: horizon (1 or 5)
    input = [actions; reshape(simmed_model_output.rewards.', 1, []); repmat(1:9, 1, 40)]; 
    input = input(:, ~isnan(input(1,:))); % remove NaN values
    input = [input; repmat(repelem([1, 5], [5, 9]), 1, 20)];   % create a row to indicate Horizon
    input = [NaN(4,1), input]; % make the first column NaN values because participants choose a bandit before learning
    input(:,find(input(3,:) == 1)-1) = NaN; % remove the result of the last free choice for each game because it's meaningless
    u = input(:,1:end-1);

    fprintf( 'Running VB to fit simulated behavior! \n' );

    [simfit_posterior, simfit_out] = VBA_NLStateSpaceModel(y, u, f_fname, g_fname, dim, options);
    model_output.simfit_out = simfit_out;

    if ~isempty(MDP.observation_params)
        phi_means_struct = cell2struct(num2cell(simfit_posterior.muPhi), MDP.observation_params);
        posterior_means_phi = transform_params_SM("untransform", phi_means_struct,MDP.observation_params); 
        phi_fields = fieldnames(posterior_means_phi);
        for i = 1:length(phi_fields)
            fit_info.(['simfit_' phi_fields{i}]) = posterior_means_phi.(phi_fields{i});
        end
    end

    if ~isempty(MDP.evolution_params)
        theta_means_struct = cell2struct(num2cell(simfit_posterior.muTheta), MDP.evolution_params);
        posterior_means_theta = transform_params_SM("untransform", theta_means_struct,MDP.evolution_params); 
        theta_fields = fieldnames(posterior_means_theta);
        for i = 1:length(theta_fields)
            fit_info.(['simfit_' theta_fields{i}]) = posterior_means_theta.(theta_fields{i});
        end
    end
    
 end