function model_output = model_SM_KF_all_choices(params, actions, rewards, mdp)
%     # This model has:
%     #   Kalman filter inference
%     #   Info bonus
% 
%     #   spatial bias is in this one and it can vary by 
% 
%     # no choice kernel
%     # inference is constant across horizon and uncertainty but can vary by 
%     # "condition".  Condition can be anything e.g. TMS or losses etc ...
%     
%     # two types of condition:
%     #   * inference fixed - e.g. horizon, uncertainty - C1, nC1
%     #   * inference varies - e.g. TMS, losses - C2, nC2
% 
%     # hyperpriors =========================================================
% 
%     # inference does not vary by condition 1, but can by condition 2
%     # note, always use j to refer to condition 2


% note that mu2 == right bandit ==  c=2 == free choice = 1

    dbstop if error;
    G = mdp.G; % num of games

    sigma_d = params.sigma_d;
    bias = params.side_bias;
    
    %%% FIT BEHAVIOR
    action_probs = nan(G,9);
    pred_errors = nan(G,10);
    pred_errors_alpha = nan(G,9);
    exp_vals = nan(G,10);
    alpha = nan(G,10);
    for g=1:G  % loop over games
        % values
        mu1 = [50 nan nan nan nan nan nan nan nan];
        mu2 = [50 nan nan nan nan nan nan nan nan];

        % learning rates 
        alpha1 = nan(1,9); 
        alpha2 = nan(1,9); 

        % standard deviations of bandits starts out at 0
        sigma1 = zeros(1,9); 
        sigma2 = zeros(1,9); 
        
        sigma_r1 = zeros(1,9); 
        sigma_r2 = zeros(1,9); 
        
        num_choices = sum(~isnan(actions(g,:)));

        for t=1:num_choices  % loop over forced-choice trials
            if t >= 5
                % compute UCB
                if mdp.C1(g)==1
                    T = 1;
                    Y = 1;
                else
                    T = 1+params.info_bonus;
                    Y = 1+params.random_exp;
                end
                UCB1 = mu1(t) + (2*log(T)/(sum(actions(1,1:t-1) == 1)*params.outcome_informativeness))^.5 + bias;
                UCB2 = mu2(t) + (2*log(T)/(sum(actions(1,1:t-1) == 2)*params.outcome_informativeness))^.5;

                % total uncertainty = add variance of both arms and then square root 
                % total uncertainty
                total_uncertainty = (sigma1(t)^2 + sigma2(t)^2)^.5;
                
                decision_noise = total_uncertainty+2*log(Y);
                
                % probability of choosing bandit 1
                p = exp(UCB1 / decision_noise) / (exp(UCB1 / decision_noise) + exp(UCB2 / decision_noise));
                action_probs(g,t) = mod(actions(g,t),2)*p + (1-mod(actions(g,t),2))*(1-p);
            end
            
            outcomes_bandit1 = rewards(g,actions(g,1:t) == 1);
            outcomes_bandit2 = rewards(g,actions(g,1:t) == 2);
                
                
            sigma_r1(t) = std(outcomes_bandit1);
            sigma_r2(t) = std(outcomes_bandit2);
            % left bandit choice so mu1 updates
            if (actions(g,t) == 1) 
                % update sigma and LR
                temp = 1/(sigma1(t)^2 + sigma_d^2) + 1/(sigma_r1(t)^2);
                sigma1(t+1) = (1/temp)^.5;
                alpha1(t) = (sigma1(t+1)+eps)/(sigma_r1(t)+eps);
                if ~isempty(outcomes_bandit2)   
                    temp = sigma2(t)^2 + sigma_d^2;
                    sigma2(t+1) = temp^.5; 
                end          
                exp_vals(g,t) = mu1(t);
                pred_errors(g,t) = (rewards(g,t) - exp_vals(g,t));
                alpha(g,t) = alpha1(t);
                pred_errors_alpha(g,t) = alpha1(t) * pred_errors(g,t);
                mu1(t+1) = mu1(t) + pred_errors_alpha(g,t);
                mu2(t+1) = mu2(t); 
            else % right bandit choice so mu2 updates
                % update LR
                temp = 1/(sigma2(t)^2 + sigma_d^2) + 1/(sigma_r2(t)^2);
                sigma2(t+1) = (1/temp)^.5;
                alpha2(t) = (sigma2(t+1)+eps)/(sigma_r2(t)+eps); % eps keeps first LR == 1
                if ~isempty(outcomes_bandit1)   
                    temp = sigma1(t)^2 + sigma_d^2;
                    sigma1(t+1) = temp^.5; 
                end
                exp_vals(g,t) = mu1(t);
                pred_errors(g,t) = (rewards(g,t) - exp_vals(g,t));
                alpha(g,t) = alpha2(t);
                pred_errors_alpha(g,t) = alpha2(t) * pred_errors(g,t);
                mu2(t+1) = mu2(t) + pred_errors_alpha(g,t);
                mu1(t+1) = mu1(t); 
            end

        end
    end
    
    % SIMULATE BEHAVIOR
    sim_pred_errors = nan(G,10);
    sim_pred_errors_alpha = nan(G,9);
    sim_exp_vals = nan(G,10);
    sim_alpha = nan(G,10);
    simmed_free_choices = nan(G,9);
    simmed_rewards = rewards;
    for g=1:G  % loop over games
        % values
        mu1 = [50 nan nan nan nan nan nan nan nan];
        mu2 = [50 nan nan nan nan nan nan nan nan];

        % learning rates 
        alpha1 = nan(1,9); 
        alpha2 = nan(1,9); 

        % standard deviations of bandits starts out at 0
        sigma1 = zeros(1,9); 
        sigma2 = zeros(1,9); 
        
        sigma_r1 = zeros(1,9); 
        sigma_r2 = zeros(1,9); 
        
        num_choices = sum(~isnan(actions(g,:)));

        for t=1:num_choices  % loop over forced-choice trials
            simmed_free_choices(g,t) = actions(g,t);
            if t >= 5
                
                % compute UCB
                if mdp.C1(g)==1
                    T = 1;
                    Y = 1;
                else
                    T = 1+params.info_bonus;
                    Y = 1+params.random_exp;
                end
                UCB1 = mu1(t) + (2*log(T)/(sum(actions(1,1:t-1) == 1)*params.outcome_informativeness))^.5 + bias;
                UCB2 = mu2(t) + (2*log(T)/(sum(actions(1,1:t-1) == 2)*params.outcome_informativeness))^.5;

                % total uncertainty = add variance of both arms and then square root 
                % total uncertainty
                total_uncertainty = (sigma1(t)^2 + sigma2(t)^2)^.5;

                decision_noise = total_uncertainty+2*log(Y);

                % probability of choosing bandit 1
                p = exp(UCB1 / decision_noise) / (exp(UCB1 / decision_noise) + exp(UCB2 / decision_noise));
                
                % simulate behavior
                u = rand(1,1);
                if u <= p
                    simmed_free_choices(g,t) = 1;
                else
                    simmed_free_choices(g,t) = 2;
                end
                % simulate outcomes according to schedule
                if simmed_free_choices(g,t) == 1
                	simmed_rewards(g,t) = mdp.bandit1_schedule(g,t);
                else
                    simmed_rewards(g,t) = mdp.bandit2_schedule(g,t);
                end
            end
            
            outcomes_bandit1 = simmed_rewards(g,simmed_free_choices(g,1:t) == 1);
            outcomes_bandit2 = simmed_rewards(g,simmed_free_choices(g,1:t) == 2);
            
            % left bandit choice so mu1 updates
            if (simmed_free_choices(g,t) == 1) 
                % update sigma and LR
                temp = 1/(sigma1(t)^2 + sigma_d^2) + 1/(sigma_r1(t)^2);
                sigma1(t+1) = (1/temp)^.5;
                alpha1(t) = (sigma1(t+1)+eps)/(sigma_r1(t)+eps);
                if ~isempty(outcomes_bandit2)   
                    temp = sigma2(t)^2 + sigma_d^2;
                    sigma2(t+1) = temp^.5; 
                end          
                sim_exp_vals(g,t) = mu1(t);
                sim_pred_errors(g,t) = (rewards(g,t) - sim_exp_vals(g,t));
                alpha(g,t) = alpha1(t);
                sim_pred_errors_alpha(g,t) = alpha1(t) * sim_pred_errors(g,t);
                mu1(t+1) = mu1(t) + sim_pred_errors_alpha(g,t);
                mu2(t+1) = mu2(t); 
            else % right bandit choice so mu2 updates
                % update LR
                temp = 1/(sigma2(t)^2 + sigma_d^2) + 1/(sigma_r2(t)^2);
                sigma2(t+1) = (1/temp)^.5;
                alpha2(t) = (sigma2(t+1)+eps)/(sigma_r2(t)+eps); % eps keeps first LR == 1
                if ~isempty(outcomes_bandit1)   
                    temp = sigma1(t)^2 + sigma_d^2;
                    sigma1(t+1) = temp^.5; 
                end
                sim_exp_vals(g,t) = mu1(t);
                sim_pred_errors(g,t) = (rewards(g,t) - sim_exp_vals(g,t));
                alpha(g,t) = alpha2(t);
                sim_pred_errors_alpha(g,t) = alpha2(t) * sim_pred_errors(g,t);
                mu2(t+1) = mu2(t) + sim_pred_errors_alpha(g,t);
                mu1(t+1) = mu1(t); 
            end


        end
        
    end
    
    
    model_output.action_probs = action_probs;
    model_output.exp_vals = sim_exp_vals;
    model_output.pred_errors = sim_pred_errors;
    model_output.pred_errors_alpha = pred_errors_alpha;
    model_output.alpha = sim_alpha;
    model_output.simmed_free_choices = simmed_free_choices;
    model_output.simmed_rewards = simmed_rewards;

end