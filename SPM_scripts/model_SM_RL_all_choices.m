function model_output = model_SM_RL_all_choices(params, actions, rewards, mdp, sim)
% hi toru
% note that mu2 == right bandit ==  c=2 == free choice = 1

    dbstop if error;
    G = mdp.G; % num of games

    side_bias = params.side_bias;
    noise_learning_rate = params.noise_learning_rate;
    baseline_info_bonus = params.baseline_info_bonus;
    baseline_noise = params.baseline_noise;
    initial_mu = params.initial_mu;
    reward_sensitivity = params.reward_sensitivity;

    param_names = fieldnames(params);


    
    
    %%% FIT BEHAVIOR
    action_probs = nan(G,9);
    pred_errors = nan(G,10);
    pred_errors_alpha = nan(G,9);
    exp_vals = nan(G,9);
    alpha = nan(G,10);
    for g=1:G  % loop over games
        % horizon is 1
        if mdp.C1(g)==1
            T = 1;
            Y = 1;
        else
            % horizon is 5
            if any(strcmp('DE_RE_horizon', param_names))
                T = 1+params.DE_RE_horizon;
                Y = 1+params.DE_RE_horizon;
            else
                T = 1+params.info_bonus;
                Y = 1+params.random_exp;                    
            end
        end
        mu1 = [initial_mu nan nan nan nan nan nan nan nan];
        mu2 = [initial_mu nan nan nan nan nan nan nan nan];
        noise = [baseline_noise * Y nan nan nan nan nan nan nan nan]; 
        if any(strcmp('associability_weight', param_names))
            associability1 = [params.initial_associability nan nan nan nan nan nan nan nan];
            associability2 = [params.initial_associability nan nan nan nan nan nan nan nan];
        end
        
        num_choices = sum(~isnan(actions(g,:)));

        for t=1:num_choices  % loop over forced-choice trials

            if t >= 5

               
               %decision = 1/(1+exp(Q1-Q2)/noise))
                deltaR = mu1(t) - mu2(t);
                deltaI = (sum(actions(g,1:t-1) == 2) - sum(actions(g,1:t-1) == 1)) * baseline_info_bonus * T;
                               
                
                % probability of choosing bandit 1
                p = 1 / (1 + exp(-(deltaR+deltaI+side_bias)/(noise(t))));
                
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
                exp_vals(g,t) = mu1(t);
                pred_errors(g,t) = (reward_sensitivity*rewards(g,t)) - exp_vals(g,t);
                if any(strcmp('associability_weight', param_names))
                    alpha(g,t) = params.learning_rate * associability1(t);
                    associability1(t+1) = (1 - params.associability_weight)*associability1(t) + params.associability_weight*abs(pred_errors(g,t));
                    associability2(t+1) = associability2(t);
                elseif any(strcmp('learning_rate_pos', param_names)) && any(strcmp('learning_rate_neg', param_names))
                    if pred_errors(g,t) > 0
                        alpha(g,t) = params.learning_rate_pos;
                    else
                        alpha(g,t) = params.learning_rate_neg;
                    end
                else 
                    alpha(g,t) = params.learning_rate;
                end
                pred_errors_alpha(g,t) = alpha(g,t) * pred_errors(g,t);
                mu1(t+1) = mu1(t) + pred_errors_alpha(g,t);
                mu2(t+1) = mu2(t); 
                

                
            else % right bandit choice so mu2 updates
                exp_vals(g,t) = mu2(t);
                pred_errors(g,t) = (reward_sensitivity*rewards(g,t)) - exp_vals(g,t);
                if any(strcmp('associability_weight', param_names))
                    alpha(g,t) = params.learning_rate * associability1(t);
                    associability1(t+1) = (1 - params.associability_weight)*associability1(t) + params.associability_weight*abs(pred_errors(g,t));
                    associability2(t+1) = associability2(t);
                elseif any(strcmp('learning_rate_pos', param_names)) && any(strcmp('learning_rate_neg', param_names))
                    if pred_errors(g,t) > 0
                        alpha(g,t) = params.learning_rate_pos;
                    else
                        alpha(g,t) = params.learning_rate_neg;
                    end
                else 
                    alpha(g,t) = params.learning_rate;
                end
                pred_errors_alpha(g,t) = alpha(g,t) * pred_errors(g,t);
                mu2(t+1) = mu2(t) + pred_errors_alpha(g,t);
                mu1(t+1) = mu1(t); 
            end
            
            % update noise with softplus function. should become more deterministic with wins.
            noise_raw = noise(t) - noise_learning_rate*(pred_errors(g,t));
            noise(t+1) = log(1+exp(noise_raw));
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
