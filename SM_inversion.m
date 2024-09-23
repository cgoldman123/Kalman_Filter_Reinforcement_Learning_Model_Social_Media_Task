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
prior_variance = 2^-2;

for i = 1:length(DCM.field)
    field = DCM.field{i};
    if ALL
        pE.(field) = zeros(size(param));
        pC{i,i}    = diag(param);
    else
        % transform the parameters that we fit
        if ismember(field, {'alpha_start', 'alpha_inf'})
            pE.(field) = log(DCM.params.(field)/(1-DCM.params.(field)));  % bound between 0 and 1
            pC{i,i}    = prior_variance;
        elseif ismember(field, {'dec_noise_h1_13', 'dec_noise_h5_13'})
            pE.(field) = log(DCM.params.(field));               % in log-space (to keep positive)
            pC{i,i}    = prior_variance;  
        elseif ismember(field,{'info_bonus_h1', 'info_bonus_h5','side_bias_h1', 'side_bias_h5'})
            pE.(field) = DCM.params.(field); 
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
        if ismember(field{i},{'alpha_start', 'alpha_inf'})
            params.(field{i}) = 1/(1+exp(-P.(field{i})));
        elseif ismember(field{i},{'alpha_start', 'alpha_inf', 'dec_noise_h1_13', 'dec_noise_h5_13'})
            params.(field{i}) = exp(P.(field{i}));
        else
            params.(field{i}) = P.(field{i});
        end
    end

    free_choices = U.c5(1,:);
    rewards = squeeze(U.r(1,:,:))';

    mdp.horizon_sequence = U.C1(1,:);
    mdp.forced_choices = squeeze(U.a(1,:,:))';
    mdp.right_info = squeeze(U.dI(1,:,:));
    mdp.T = 4; % num forced choices
    mdp.G = 40; % game length
        
    % note that mu2 == right bandit ==  c=2 == free choice = 1


    model_output = model_SM_KF_all_choices(params,free_choices, rewards,mdp);
    log_probs = log(model_output.action_probs);
    log_probs(isnan(log_probs)) = eps; % Replace NaN in log output with eps for summing
    L = sum(log_probs, 'all');




fprintf('LL: %f \n',L)


