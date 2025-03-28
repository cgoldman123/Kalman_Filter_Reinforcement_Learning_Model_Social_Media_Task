function  [fx] = f_KF_SIGMA4 (x, P, u, in)
% Kalman filter learning rule for the Horizon Task
    % Input variable u
    % row 1: actions
    % row 2: rewards
    % row 3: timestep within a game (1 to 9)
    % row 4: horizon (1 or 5)

    % State variable x
    % row 1: left bandit mean
    % row 2: right bandit mean
    % row 3: left bandit sigma
    % row 4: right bandit sigma

% initialize output variable
fx = nan(length(x),1);

% Pull out parameters from vector P
for i = 1:numel(in.MDP.evolution_params)
    field_name = in.MDP.evolution_params{i}; % Extract the field name
    params.(field_name) = P(i); % Assign the value from P
end
% Transform parameters back to native space
if exist("params", "var")
    retrans_params = transform_params_SM("untransform", params,in.MDP.evolution_params);
else
    retrans_params = [];
end
% If retrans_params does not contain a parameter that is needed (i.e.,
% a parameter not fit), add it
for f = fieldnames(in.MDP.params)'
    if ~isfield(retrans_params, f{1})
        retrans_params.(f{1}) = in.MDP.params.(f{1});
    end
end

sigma_d = retrans_params.sigma_d;
sigma_r = retrans_params.sigma_r;
sigma1 = x(3);
sigma2 = x(4);

% left bandit choice so mu1 updates
if ~u(1)
    % save relative uncertainty of choice
    %relative_uncertainty_of_choice(g,t) = sigma1(g,t) - sigma2(g,t);
    % update sigma and LR
    temp = 1/(sigma1^2 + sigma_d^2) + 1/(sigma_r^2);
    fx(3) = (1/temp)^.5;
    %change_in_uncertainty_after_choice(g,t) = sigma1(g,t+1) - sigma1(g,t);
    alpha1 = (fx(3)/(sigma_r))^2; 
    
    temp = sigma2^2 + sigma_d^2;
    fx(4) = temp^.5; 

    %exp_vals(g,t) = mu1(t);
    pred_error = (in.MDP.params.reward_sensitivity*u(2)) - x(1);
    %alpha(g,t) = alpha1(t);
    pred_errors_alpha = alpha1 * pred_error;
    fx(1) = x(1) + pred_errors_alpha;
    fx(2) = x(2); 
elseif u(1) % right bandit choice so mu2 updates
    % save relative uncertainty of choice
    %relative_uncertainty_of_choice(g,t) = sigma2(g,t) - sigma1(g,t);
    % update LR
    temp = 1/(sigma2^2 + sigma_d^2) + 1/(sigma_r^2);
    fx(4) = (1/temp)^.5;
    %change_in_uncertainty_after_choice(g,t) = sigma2(g,t+1) - sigma2(g,t);
    alpha2 = (fx(4)/(sigma_r))^2; 
     
    temp = sigma1^2 + sigma_d^2;
    fx(3) = temp^.5; 
    
    %exp_vals(g,t) = mu2(t);
    pred_error = (in.MDP.params.reward_sensitivity*u(2)) - x(2);
    %alpha(g,t) = alpha2(t);
    pred_errors_alpha = alpha2 * pred_error;
    fx(2) = x(2) + pred_errors_alpha;
    fx(1) = x(1); 
end


