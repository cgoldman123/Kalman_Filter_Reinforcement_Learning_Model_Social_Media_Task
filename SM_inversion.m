% Model inversion script
function [DCM] = SM_inversion(DCM)

% MDP inversion using Variational Bayes
% FORMAT [DCM] = spm_dcm_mdp(DCM)

% If simulating - comment out section on line 196
% If not simulating - specify subject data file in this section 

%
% Expects:
%--------------------------------------------------------------------------
% DCM.MDP   % MDP structure specifying a generative model
% DCM.field % parameter (field) names to optimise
% DCM.U     % cell array of outcomes (stimuli)
% DCM.Y     % cell array of responses (action)
%
% Returns:
%--------------------------------------------------------------------------
% DCM.M     % generative model (DCM)
% DCM.Ep    % Conditional means (structure)
% DCM.Cp    % Conditional covariances
% DCM.F     % (negative) Free-energy bound on log evidence
% 
% This routine inverts (cell arrays of) trials specified in terms of the
% stimuli or outcomes and subsequent choices or responses. It first
% computes the prior expectations (and covariances) of the free parameters
% specified by DCM.field. These parameters are log scaling parameters that
% are applied to the fields of DCM.MDP. 
%
% If there is no learning implicit in multi-trial games, only unique trials
% (as specified by the stimuli), are used to generate (subjective)
% posteriors over choice or action. Otherwise, all trials are used in the
% order specified. The ensuing posterior probabilities over choices are
% used with the specified choices or actions to evaluate their log
% probability. This is used to optimise the MDP (hyper) parameters in
% DCM.field using variational Laplace (with numerical evaluation of the
% curvature).
%
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_dcm_mdp.m 7120 2017-06-20 11:30:30Z spm $

% OPTIONS
%--------------------------------------------------------------------------
ALL = false;

% prior expectations and covariance
%--------------------------------------------------------------------------
prior_variance = 2;

for i = 1:length(DCM.field)
    field = DCM.field{i};
    if ALL
        pE.(field) = zeros(size(param));
        pC{i,i}    = diag(param);
    else
        % transform the parameters that we fit
        if ismember(field, {'learning_rate', 'learning_rate_pos', 'learning_rate_neg', 'noise_learning_rate', 'alpha_start', 'alpha_inf', 'associability_weight', ...
                'starting_bias_baseline'})
            pE.(field) = log(DCM.params.(field)/(1-DCM.params.(field)));  % bound between 0 and 1
            pC{i,i}    = prior_variance;
        elseif ismember(field, {'h1_dec_noise', 'h5_baseline_dec_noise', 'h5_slope_dec_noise', ...
                'initial_sigma', 'initial_sigma_r', 'initial_mu', 'initial_associability', ...
                'drift_action_prob_mod', 'drift_reward_diff_mod', 'drift_UCB_diff_mod',...
                'starting_bias_action_prob_mod', 'starting_bias_reward_diff_mod', 'starting_bias_UCB_diff_mod',...
                'decision_thresh_baseline', 'decision_thresh_action_prob_mod', 'decision_thresh_reward_diff_mod', 'decision_thresh_UCB_diff_mod', ...
                'outcome_informativeness', 'sigma_d', 'info_bonus', 'random_exp', 'baseline_noise', ...
                'sigma_r', 'reward_sensitivity', 'DE_RE_horizon'})
            pE.(field) = log(DCM.params.(field));               % in log-space (to keep positive)
            pC{i,i}    = prior_variance;  
        elseif ismember(field, {'h5_baseline_info_bonus', 'h5_slope_info_bonus', 'h1_info_bonus', 'baseline_info_bonus',...
                'side_bias', 'side_bias_h1', 'side_bias_h5', ...
                'drift_baseline', 'drift'})
            pE.(field) = DCM.params.(field); 
            pC{i,i}    = prior_variance;
        elseif any(strcmp(field,{'nondecision_time'})) % bound between .1 and .3
            pE.(field) =  -log((0.3 - 0.1) ./ (DCM.params.(field) - 0.1) - 1);             
            pC{i,i}    = prior_variance;      
        else
            disp(field);
            error("Param not properly transformed");
        end
    end
end

pC      = spm_cat(pC);

% model specification
%--------------------------------------------------------------------------
M.L     = @(P,M,U,Y)spm_mdp_L(P,M,U,Y);  % log-likelihood function
M.pE    = pE;                            % prior means (parameters)
M.pC    = pC;                            % prior variance (parameters)
M.params = DCM.params;                   % includes fixed and fitted params
M.model = DCM.model;

% Variational Laplace
%--------------------------------------------------------------------------
[Ep,Cp,F] = spm_nlsi_Newton(M,DCM.datastruct,DCM.datastruct);

% Store posterior densities and log evidnce (free energy)
%--------------------------------------------------------------------------
DCM.M   = M;
DCM.Ep  = Ep;
DCM.Cp  = Cp;
DCM.F   = F;


return

function L = spm_mdp_L(P,M,U,Y)
    % log-likelihood function
    % FORMAT L = spm_mdp_L(P,M,U,Y)
    % P    - parameter structure
    % M    - generative model
    % U    - observations
    % Y    - actions
    %__________________________________________________________________________

    if ~isstruct(P); P = spm_unvec(P,M.pE); end

    % multiply parameters in MDP
    %--------------------------------------------------------------------------
    params   = M.params; % includes fitted and fixed params. Write over fitted params below. 
    field = fieldnames(M.pE);
    for i = 1:length(field)
        if ismember(field{i},{'learning_rate', 'learning_rate_pos', 'learning_rate_neg', 'noise_learning_rate', 'alpha_start', 'alpha_inf', 'associability_weight', ...
                'starting_bias_baseline'})
            params.(field{i}) = 1/(1+exp(-P.(field{i})));
        elseif ismember(field{i},{'h1_dec_noise', 'h5_baseline_dec_noise', 'h5_slope_dec_noise', ...
                'initial_sigma', 'initial_sigma_r', 'initial_mu', 'initial_associability', ...
                'drift_action_prob_mod', 'drift_reward_diff_mod', 'drift_UCB_diff_mod',...
                'starting_bias_action_prob_mod', 'starting_bias_reward_diff_mod', 'starting_bias_UCB_diff_mod',...
                'decision_thresh_baseline', 'decision_thresh_action_prob_mod', 'decision_thresh_reward_diff_mod', 'decision_thresh_UCB_diff_mod', ...
                'outcome_informativeness', 'sigma_d', 'info_bonus', 'random_exp', 'baseline_noise', ...
                'sigma_r', 'reward_sensitivity', 'DE_RE_horizon'})
            params.(field{i}) = exp(P.(field{i}));
        elseif ismember(field{i},{'h5_baseline_info_bonus', 'h5_slope_info_bonus', 'h1_info_bonus', 'baseline_info_bonus',...
                'side_bias', 'side_bias_h1', 'side_bias_h5', ...
                'drift_baseline', 'drift'})
            params.(field{i}) = P.(field{i});
        else
            error("Param not transformed properly");
        end
    end

    actions_and_rts.actions = U.actions;
    actions_and_rts.RTs = U.RTs;
    rewards = U.rewards;

    mdp = U;
        
    % note that mu2 == right bandit ==  c=2 == free choice = 1
    model_output = M.model(params,actions_and_rts, rewards,mdp, 0);

    % Fit to reaction time pdfs if DDM, fit to action probabilities if
    % choice model
    if strcmp(func2str(M.model), 'model_SM_KF_DDM_all_choices')
        log_probs = log(model_output.rt_pdf+eps);
    else
        log_probs = log(model_output.action_probs+eps);
    end
    log_probs(isnan(log_probs)) = eps; % Replace NaN in log output with eps for summing
    L = sum(log_probs, 'all');




fprintf('LL: %f \n',L)


