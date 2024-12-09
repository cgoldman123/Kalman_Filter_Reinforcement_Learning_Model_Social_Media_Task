function model_output = model_SM_KF_all_choices(params, actions, rewards, mdp, sim)
% hi toru
% note that mu2 == right bandit ==  c=2 == free choice = 1

    dbstop if error;
    G = mdp.G; % num of games

    sigma_d = params.sigma_d;
    bias = params.side_bias;
    sigma_r = params.sigma_r;
    initial_sigma = params.initial_sigma;
    baseline_info_bonus = params.baseline_info_bonus;
    baseline_noise = params.baseline_noise;
    initial_mu = params.initial_mu;
    reward_sensitivity = params.reward_sensitivity;
    param_names = fieldnames(params);

    % indicate if want one parameter to control DE/RE or keep separate
    if any(strcmp('DE_RE_horizon', param_names))
        DE_RE_horizon = params.DE_RE_horizon;
    else
        info_bonus = params.info_bonus;
        random_exp = params.random_exp;
    end
    
    %%% FIT BEHAVIOR
    action_probs = nan(G,9);
    pred_errors = nan(G,10);
    pred_errors_alpha = nan(G,9);
    exp_vals = nan(G,10);
    alpha = nan(G,10);
    for g=1:G  % loop over games
        % values
        mu1 = [initial_mu nan nan nan nan nan nan nan nan];
        mu2 = [initial_mu nan nan nan nan nan nan nan nan];

        % learning rates 
        alpha1 = nan(1,9); 
        alpha2 = nan(1,9); 

        sigma1 = nan(1,9); 
        sigma1(1) = initial_sigma;
        sigma2 = nan(1,9); 
        sigma2(1) = initial_sigma;
        
        num_choices = sum(~isnan(actions(g,:)));

        for t=1:num_choices  % loop over forced-choice trials
            if t >= 5
                % compute UCB
                % horizon is 1
                if mdp.C1(g)==1
                    T = 1;
                    Y = 1;
                else
                    % horizon is 5
                    if any(strcmp('DE_RE_horizon', param_names))
                        T = 1+DE_RE_horizon;
                        Y = 1+DE_RE_horizon;
                    else
                        T = 1+info_bonus;
                        Y = 1+random_exp;                    
                    end
                end
               
               %decision = 1/(1+exp(Q1-Q2)/noise))
                Q1 = mu1(t) - sum(actions(g,1:t-1) == 1)*baseline_info_bonus + sigma1(t)*log(T) + bias;
                Q2 = mu2(t) - sum(actions(g,1:t-1) == 2)*baseline_info_bonus + sigma2(t)*log(T);

                % total uncertainty = add variance of both arms and then square root 
                % total uncertainty
                total_uncertainty = (sigma1(t)^2 + sigma2(t)^2)^.5;
                
                decision_noise = 1+total_uncertainty*log(Y)+ baseline_noise;
                
                % probability of choosing bandit 1
                p = 1 / (1 + exp(-(Q1-Q2)/(decision_noise)));
                
                if sim
                    % simulate behavior
                    u = rand(1,1);
                    if u <= p
                        actions(g,t) = 1;
                        rewards(g,t) = mdp.bandit1_schedule(g,t);
                    else
                        actions(g,t) = 2;
                        rewards(g,t) = mdp.bandit2_schedule(g,t);
                    end
                end
                
                action_probs(g,t) = mod(actions(g,t),2)*p + (1-mod(actions(g,t),2))*(1-p);
            end
                
            
            % left bandit choice so mu1 updates
            if (actions(g,t) == 1) 
                % update sigma and LR
                temp = 1/(sigma1(t)^2 + sigma_d^2) + 1/(sigma_r^2);
                sigma1(t+1) = (1/temp)^.5;
                alpha1(t) = (sigma1(t+1)/(sigma_r))^2; 
                
                temp = sigma2(t)^2 + sigma_d^2;
                sigma2(t+1) = temp^.5; 
        
                exp_vals(g,t) = mu1(t);
                pred_errors(g,t) = (reward_sensitivity*rewards(g,t)) - exp_vals(g,t);
                alpha(g,t) = alpha1(t);
                pred_errors_alpha(g,t) = alpha1(t) * pred_errors(g,t);
                mu1(t+1) = mu1(t) + pred_errors_alpha(g,t);
                mu2(t+1) = mu2(t); 
            else % right bandit choice so mu2 updates
                % update LR
                temp = 1/(sigma2(t)^2 + sigma_d^2) + 1/(sigma_r^2);
                sigma2(t+1) = (1/temp)^.5;
                alpha2(t) = (sigma2(t+1)/(sigma_r))^2; 
                 
                temp = sigma1(t)^2 + sigma_d^2;
                sigma1(t+1) = temp^.5; 
                
                exp_vals(g,t) = mu2(t);
                pred_errors(g,t) = (reward_sensitivity*rewards(g,t)) - exp_vals(g,t);
                alpha(g,t) = alpha2(t);
                pred_errors_alpha(g,t) = alpha2(t) * pred_errors(g,t);
                mu2(t+1) = mu2(t) + pred_errors_alpha(g,t);
                mu1(t+1) = mu1(t); 
            end

        end
    end

    
    
    model_output.action_probs = action_probs;
    model_output.exp_vals = exp_vals;
    model_output.pred_errors = pred_errors;
    model_output.pred_errors_alpha = pred_errors_alpha;
    model_output.alpha = alpha;
    model_output.actions = actions;
    model_output.rewards = rewards;

end