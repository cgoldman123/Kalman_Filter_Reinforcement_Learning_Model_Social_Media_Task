function gx = g_KF_SIGMA4_logistic_no_latent_state_learning(x, P, u, in)    
    dbstop if error;
        % Pull out parameters from vector P
    for i = 1:numel(in.MDP.observation_params)
        field_name = in.MDP.observation_params{i}; % Extract the field name
        params.(field_name) = P(i); % Assign the value from P
    end
    % Transform parameters back to native space
    if exist("params","var")
        retrans_params = transform_params_SM("untransform", params,in.MDP.observation_params); 
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

    G = in.G; % num of games    
    
    % initialize params
    sigma_d = retrans_params.sigma_d;
    side_bias = retrans_params.side_bias;
    sigma_r = retrans_params.sigma_r;
    initial_sigma = retrans_params.initial_sigma;
    initial_mu = retrans_params.initial_mu;
    reward_sensitivity = retrans_params.reward_sensitivity;   
    baseline_info_bonus = retrans_params.baseline_info_bonus;
    directed_exp = retrans_params.directed_exp;
    random_exp = retrans_params.random_exp;
    baseline_noise = retrans_params.baseline_noise;
    
   
    
    % initialize variables
    actions = in.actions;
    action_probs = nan(G,9);
    prob_choose_bandit2 = nan(G,9);
    model_acc = nan(G,9);
    
    pred_errors = nan(G,10);
    pred_errors_alpha = nan(G,9);
    exp_vals = nan(G,10);
    alpha = nan(G,10);
    sigma1 = [initial_sigma * ones(G,1), zeros(G,8)];
    sigma2 = [initial_sigma * ones(G,1), zeros(G,8)];
    total_uncertainty = nan(G,9);
    relative_uncertainty_of_choice = nan(G,9);
    change_in_uncertainty_after_choice = nan(G,9);


    
    for g=1:G  % loop over games
        % values
        mu1 = [initial_mu nan nan nan nan nan nan nan nan];
        mu2 = [initial_mu nan nan nan nan nan nan nan nan];

        % learning rates 
        alpha1 = nan(1,9); 
        alpha2 = nan(1,9); 
        
        num_choices = sum(~isnan(in.rewards(g,:))); 


        for t=1:num_choices  % loop over forced-choice trials
            if t >= 5
                if in.C1(g)==1 % horizon is 1
                    T = 0;
                    Y = 1;
                else % horizon is 5
                    T = directed_exp;
                    Y = random_exp;                    
                end
                
                reward_diff = mu1(t) - mu2(t);
                z = .5; % hyperparam controlling steepness of curve
                
                 % % Exponential descent
                 info_bonus_bandit1 = sigma1(g,t)*baseline_info_bonus + sigma1(g,t)*T*(exp(-z*(t-5))-exp(-4*z))/(1-exp(-4*z));
                 info_bonus_bandit2 = sigma2(g,t)*baseline_info_bonus + sigma2(g,t)*T*(exp(-z*(t-5))-exp(-4*z))/(1-exp(-4*z));

                 % Linear descent
                 % info_bonus_bandit1 = sigma1(g,t)*baseline_info_bonus + sigma1(g,t)*T*((9 - t)/4);
                 % info_bonus_bandit2 = sigma2(g,t)*baseline_info_bonus + sigma2(g,t)*T*((9 - t)/4);

                 info_diff = info_bonus_bandit1 - info_bonus_bandit2;
                


                % total uncertainty is variance of both arms
                total_uncertainty(g,t) = (sigma1(g,t)^2 + sigma2(g,t)^2)^.5;
                
                 % % Exponential descent
                 RE = Y + ((1 - Y) * (1 - exp(-z * (t - 5))) / (1 - exp(-4 * z)));

                 % Linear descent
                 % RE = Y * ((9 - t)/4);
                
                decision_noise = total_uncertainty(g,t)*baseline_noise*RE;


                % probability of choosing bandit 1
                p = 1 / (1 + exp(-(reward_diff+info_diff+side_bias)/(decision_noise)));
                
                prob_choose_bandit2(g,t) = 1-p;
            
                % simulate behavior
                if in.sim
                    u = rand(1,1);
                    if u <= p
                        actions(g,t) = 1;
                        in.rewards(g,t) = in.bandit1_schedule(g,t);
                    else
                        actions(g,t) = 2;
                        in.rewards(g,t) = in.bandit2_schedule(g,t);
                    end
                end
                action_probs(g,t) = mod(actions(g,t),2)*p + (1-mod(actions(g,t),2))*(1-p);
                model_acc(g,t) =  action_probs(g,t) > .5;     
                
            end
                
            
            % left bandit choice so mu1 updates
            if (actions(g,t) == 1) 
                % save relative uncertainty of choice
                relative_uncertainty_of_choice(g,t) = sigma1(g,t) - sigma2(g,t);

                % update sigma and LR
                temp = 1/(sigma1(g,t)^2 + sigma_d^2) + 1/(sigma_r^2);
                sigma1(g,t+1) = (1/temp)^.5;
                change_in_uncertainty_after_choice(g,t) = sigma1(g,t+1) - sigma1(g,t);
                alpha1(t) = (sigma1(g,t+1)/(sigma_r))^2; 
                
                temp = sigma2(g,t)^2 + sigma_d^2;
                sigma2(g,t+1) = temp^.5; 
        
                exp_vals(g,t) = mu1(t);
                pred_errors(g,t) = (reward_sensitivity*in.rewards(g,t)) - exp_vals(g,t);
                alpha(g,t) = alpha1(t);
                pred_errors_alpha(g,t) = alpha1(t) * pred_errors(g,t);
                mu1(t+1) = mu1(t) + pred_errors_alpha(g,t);
                mu2(t+1) = mu2(t); 
            else % right bandit choice so mu2 updates
                % save relative uncertainty of choice
                relative_uncertainty_of_choice(g,t) = sigma2(g,t) - sigma1(g,t);
                % update LR
                temp = 1/(sigma2(g,t)^2 + sigma_d^2) + 1/(sigma_r^2);
                sigma2(g,t+1) = (1/temp)^.5;
                change_in_uncertainty_after_choice(g,t) = sigma2(g,t+1) - sigma2(g,t);
                alpha2(t) = (sigma2(g,t+1)/(sigma_r))^2; 
                 
                temp = sigma1(g,t)^2 + sigma_d^2;
                sigma1(g,t+1) = temp^.5; 
                
                exp_vals(g,t) = mu2(t);
                pred_errors(g,t) = (reward_sensitivity*in.rewards(g,t)) - exp_vals(g,t);
                alpha(g,t) = alpha2(t);
                pred_errors_alpha(g,t) = alpha2(t) * pred_errors(g,t);
                mu2(t+1) = mu2(t) + pred_errors_alpha(g,t);
                mu1(t+1) = mu1(t); 
            end


        end
    end

    if in.get_summary_stats
        % If getting summary statistics, store them in variable gx
        gx.action_probs = action_probs;
        gx.model_acc = model_acc;
    
        gx.exp_vals = exp_vals;
        gx.pred_errors = pred_errors;
        gx.pred_errors_alpha = pred_errors_alpha;
        gx.alpha = alpha;
        gx.sigma1 = sigma1;
        gx.sigma2 = sigma2;
        gx.actions = actions;
        gx.rewards = rewards;
        gx.relative_uncertainty_of_choice = relative_uncertainty_of_choice;
        gx.total_uncertainty = total_uncertainty;
        gx.change_in_uncertainty_after_choice = change_in_uncertainty_after_choice;
    else
        free_choices_prob_choose_bandit2 = prob_choose_bandit2(:,5:end);
        reshaped_prob_choose_bandit_2 =  [reshape(free_choices_prob_choose_bandit2.', 1, [])];
        no_nan_prob_choose_bandit_2 = reshaped_prob_choose_bandit_2(~isnan(reshaped_prob_choose_bandit_2)); % remove NaN values
        gx = no_nan_prob_choose_bandit_2(u(1));
    end

end