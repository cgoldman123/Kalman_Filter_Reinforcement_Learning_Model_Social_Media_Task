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
prior_variance = 1/4;

global params_old;
params_old = [];

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
        elseif ismember(field, {'h1_dec_noise', 'h5_dec_noise', 'h5_baseline_dec_noise', 'h5_slope_dec_noise', ...
                'initial_sigma', 'initial_sigma_r', 'initial_mu', 'initial_associability', ...
                'drift_action_prob_mod', 'drift_reward_diff_mod', 'drift_UCB_diff_mod',...
                'starting_bias_action_prob_mod', 'starting_bias_reward_diff_mod', 'starting_bias_UCB_diff_mod',...
                'decision_thresh_action_prob_mod', 'decision_thresh_reward_diff_mod', 'decision_thresh_UCB_diff_mod', 'decision_thresh_decision_noise_mod' ...
                'outcome_informativeness', ...
                'reward_sensitivity', 'DE_RE_horizon'})
            pE.(field) = log(DCM.params.(field));               % in log-space (to keep positive)
            pC{i,i}    = prior_variance;  
        elseif ismember(field, {'h5_baseline_info_bonus', 'h5_slope_info_bonus', 'h1_info_bonus', ...
                'side_bias', 'side_bias_h1', 'side_bias_h5', 'info_bonus', 'h5_info_bonus',...
                'drift_baseline', 'drift'})
            pE.(field) = DCM.params.(field); 
            pC{i,i}    = prior_variance;
        elseif any(strcmp(field,{'nondecision_time'})) % bound between .1 and .3
            pE.(field) =  -log((0.3 - 0.1) ./ (DCM.params.(field) - 0.1) - 1);             
            pC{i,i}    = prior_variance;      
        elseif any(strcmp(field,{'decision_thresh_baseline'})) % bound greater than .5 and less than 1000
            pE.(field) =  -log((1000 - .5) ./ (DCM.params.(field) - .5) - 1);             
            pC{i,i}    = prior_variance;      
        elseif any(strcmp(field,{'sigma_d'})) % bound between 0 and 40
            pE.(field) =  -log((40) ./ (DCM.params.(field)) - 1);     
            pC{i,i}    = 1/4;                
        elseif any(strcmp(field,{'sigma_r'})) % bound between 0 and 40
            pE.(field) =  -log((40) ./ (DCM.params.(field)) - 1);     
            pC{i,i}    = 1/4;                
        elseif any(strcmp(field,{'baseline_info_bonus'})) % bound between 0 and 40
            pE.(field) = DCM.params.(field); 
            pC{i,i}    = 1/4;                
        elseif any(strcmp(field,{'directed_exp'})) 
            pE.(field) = DCM.params.(field); 
            pC{i,i}    = 1/4;                
        elseif any(strcmp(field,{'baseline_noise'})) 
            pE.(field) = log(DCM.params.(field));               % in log-space (to keep positive)
            pC{i,i}    = 1/4;                
        elseif any(strcmp(field,{'random_exp'})) % bound between 0 and 40
            pE.(field) = log(DCM.params.(field));               % in log-space (to keep positive)
            pC{i,i}    = 1/4;                 
        elseif any(strcmp(field,{'ws'})) 
            pE.(field) = log(DCM.params.(field)/(1-DCM.params.(field)));    
            pC{i,i}    = 1/4;   
        elseif any(strcmp(field,{'wd'})) %
            pE.(field) = log(DCM.params.(field)/(1-DCM.params.(field))); 
            pC{i,i}    = 1/4;   
        elseif any(strcmp(field,{'V0'})) 
            pE.(field) = DCM.params.(field);               % in log-space (to keep positive)
            pC{i,i}    = 1; 
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
    global params_old;
    
    if ~isstruct(P); P = spm_unvec(P,M.pE); end

    % multiply parameters in MDP
    %--------------------------------------------------------------------------
    params = M.params; % includes fitted and fixed params. Write over fitted params below. 
    field = fieldnames(M.pE);
    for i = 1:length(field)
        if ismember(field{i},{'learning_rate', 'learning_rate_pos', 'learning_rate_neg', 'noise_learning_rate', 'alpha_start', 'alpha_inf', 'associability_weight', ...
                'starting_bias_baseline', 'wd', 'ws'})
            params.(field{i}) = 1/(1+exp(-P.(field{i})));
        elseif ismember(field{i},{'h1_dec_noise', 'h5_dec_noise','h5_baseline_dec_noise', 'h5_slope_dec_noise', ...
                'initial_sigma', 'initial_sigma_r', 'initial_mu', 'initial_associability', ...
                'drift_action_prob_mod', 'drift_reward_diff_mod', 'drift_UCB_diff_mod',...
                'starting_bias_action_prob_mod', 'starting_bias_reward_diff_mod', 'starting_bias_UCB_diff_mod',...
                'decision_thresh_action_prob_mod', 'decision_thresh_reward_diff_mod', 'decision_thresh_UCB_diff_mod', 'decision_thresh_decision_noise_mod'...
                'outcome_informativeness', 'random_exp', 'baseline_noise', ...
                'reward_sensitivity', 'DE_RE_horizon'})
            params.(field{i}) = exp(P.(field{i}));
        elseif ismember(field{i},{'h5_baseline_info_bonus', 'h5_slope_info_bonus', 'h1_info_bonus', 'baseline_info_bonus',...
                'side_bias', 'side_bias_h1', 'side_bias_h5', 'info_bonus', 'h5_info_bonus',...
                'drift_baseline', 'drift', 'directed_exp', 'V0'})
            params.(field{i}) = P.(field{i});
        elseif ismember(field{i},{'decision_thresh_baseline'})
            params.(field{i}) = .5 + (1000 - .5) ./ (1 + exp(-P.(field{i})));     
        elseif ismember(field{i},{'sigma_d','sigma_r'})
            params.(field{i}) = (40) ./ (1 + exp(-P.(field{i})));     
        else
            error("Param not transformed properly");
        end
    end

    % Check for significant parameter changes
    if ~isempty(params_old)
        any_significant_changes = false;
        for i = 1:length(field)
            f = field{i};
            if abs(params.(f) - params_old.(f)) > 0.1
                any_significant_changes = true;
                break;
            end
        end
        
        if any_significant_changes
            fprintf('\nParameter values:\n');
            for i = 1:length(field)
                f = field{i};
                fprintf('%s: %.4f\n', f, params.(f));
            end
            fprintf('\n');
        end
    end
    
    % Update params_old for next comparison
    params_old = params;

    actions_and_rts.actions = U.actions;
    actions_and_rts.RTs = U.RTs;
    rewards = U.rewards;

    mdp = U;
        
    % note that mu2 == right bandit ==  c=2 == free choice = 1
    model_output = M.model(params,actions_and_rts, rewards,mdp, 0);

    % Fit to reaction time pdfs if DDM, fit to action probabilities if
    % choice model
    if ismember(func2str(M.model), {'model_SM_KF_DDM_all_choices', 'model_SM_KF_SIGMA_DDM_all_choices', 'model_SM_KF_SIGMA_logistic_DDM', 'model_SM_KF_SIGMA_logistic_RACING'})
        log_probs = log(model_output.rt_pdf+eps);
        summed_log_probs = sum(log_probs(~isnan(log_probs)));
        % if any log probs were NaN that should not be, throw an error.
        % Note that we expect a different number of nan values for the
        % models fitting all free choices vs first free choice.
        % Models fitting all choices:
        if ismember(func2str(M.model), {'model_SM_KF_DDM_all_choices', 'model_SM_KF_SIGMA_DDM_all_choices'})
            number_nan_log_probs = 120 - model_output.num_invalid_rts - sum(~isnan(log_probs(:)));
        % Models fitting first choice
        elseif ismember(func2str(M.model), {'model_SM_KF_SIGMA_logistic_DDM', 'model_SM_KF_SIGMA_logistic_RACING'})
            number_nan_log_probs = 40 - model_output.num_invalid_rts - sum(~isnan(log_probs(:)));
        end
        if number_nan_log_probs > 0
            error("Error! NaNs encountered in the log likelihood!")
        end
        L = summed_log_probs + number_nan_log_probs*log(realmin);

    else
        log_probs = log(model_output.action_probs+eps);
        log_probs(isnan(log_probs)) = 0; % Replace NaN in log output with 0 for summing
        L = sum(log_probs, 'all');
    end





fprintf('LL: %f \n',L)


