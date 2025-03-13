function model_output = model_SM_KF_SIGMA_all_choices(params, actions_and_rts, rewards, mdp, sim)
    dbstop if error;
    % note that mu2 == right bandit ==  c=2 == free choice = 1
    G = mdp.G; % num of games    
    
    % initialize params
    sigma_d = params.sigma_d;
    side_bias = params.side_bias;
    sigma_r = params.sigma_r;
    initial_sigma = params.initial_sigma;
    initial_mu = params.initial_mu;
    reward_sensitivity = params.reward_sensitivity;   
    baseline_info_bonus = params.baseline_info_bonus;
    directed_exp = params.directed_exp;
    random_exp = params.random_exp;
    baseline_noise = params.baseline_noise;
    
   
    
    % initialize variables
    actions = actions_and_rts.actions;
    action_probs = nan(G,9);
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
        
        num_choices = sum(~isnan(rewards(g,:))); 


        for t=1:num_choices  % loop over forced-choice trials
            if t >= 5
                if mdp.C1(g)==1 % horizon is 1
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
                
                
            
                % simulate behavior
                if sim
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
                pred_errors(g,t) = (reward_sensitivity*rewards(g,t)) - exp_vals(g,t);
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
                pred_errors(g,t) = (reward_sensitivity*rewards(g,t)) - exp_vals(g,t);
                alpha(g,t) = alpha2(t);
                pred_errors_alpha(g,t) = alpha2(t) * pred_errors(g,t);
                mu2(t+1) = mu2(t) + pred_errors_alpha(g,t);
                mu1(t+1) = mu1(t); 
            end


        end
    end

    
    model_output.action_probs = action_probs;
    model_output.model_acc = model_acc;
    
    model_output.exp_vals = exp_vals;
    model_output.pred_errors = pred_errors;
    model_output.pred_errors_alpha = pred_errors_alpha;
    model_output.alpha = alpha;
    model_output.sigma1 = sigma1;
    model_output.sigma2 = sigma2;
    model_output.actions = actions;
    model_output.rewards = rewards;
    model_output.relative_uncertainty_of_choice = relative_uncertainty_of_choice;
    model_output.total_uncertainty = total_uncertainty;
    model_output.change_in_uncertainty_after_choice = change_in_uncertainty_after_choice;


end