function model_output = model_SM_KF_SIGMA_logistic(params, actions_and_rts, rewards, mdp, sim)
    dbstop if error;
    % note that mu2 == right bandit ==  c=2 == free choice = 1
    G = mdp.G; % num of games    
    
    % initialize params
    sigma_d = params.sigma_d;
    side_bias_h1 = params.side_bias_h1;
    side_bias_h5 = params.side_bias_h5;
    sigma_r = params.sigma_r;
    initial_sigma = params.initial_sigma;
    initial_mu = params.initial_mu;
    reward_sensitivity = params.reward_sensitivity;   
    h1_info_bonus = params.h1_info_bonus;
    h5_info_bonus = params.h5_info_bonus;
    h1_dec_noise = params.h1_dec_noise;
    h5_dec_noise = params.h5_dec_noise;
    
   
    
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
        
        % if H1 Game, use H1 info bonus, decision noise, and side bias
        if mdp.C1(g) == 1
            info_bonus = h1_info_bonus;
            decision_noise = h1_dec_noise;
            side_bias = side_bias_h1;
        else
        % if H5 Game, use H5 info bonus, decision noise, and side bias
            info_bonus = h5_info_bonus;
            decision_noise = h5_dec_noise;
            side_bias = side_bias_h5;
        end



        
        for t=1:5  % loop over 4 forced choices and 1 free choice trials
            if t == 5
                % Get the reward difference
                reward_diff = mu2(t) - mu1(t);
                % Get the information difference (+1 when fewer options are
                % shown on the right, -1 when fewer options are shown on
                % left)
                info_diff = mdp.dI(g);
                % total uncertainty is variance of both arms
                % probability of choosing bandit 1
                p = 1 / (1 + exp((reward_diff+(info_diff*info_bonus)+side_bias)/(decision_noise)));
                            
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
          
            % Store total uncertainty unless someone wants it later
            total_uncertainty(g,t) = (sigma1(g,t)^2 + sigma2(g,t)^2)^.5;

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