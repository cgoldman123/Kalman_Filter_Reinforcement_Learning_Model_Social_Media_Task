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

    alpha_start = params.alpha_start;
    alpha_inf = params.alpha_inf;
    %mu0 = params.mu0; % initial value. can fix to 50
%     info_bonuses = [params.info_bonus_h1 params.info_bonus_h5];       
    decision_noises = [params.dec_noise_h1_13 params.dec_noise_h5_13];
    bias = params.side_bias;

    alpha0  = alpha_start / (1 - alpha_start) - alpha_inf^2 / (1 - alpha_inf);
    alpha_d = alpha_inf^2 / (1 - alpha_inf); 

    
    % fit sigma d, assume learning rate starts at 1
    
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
        alpha1 = [alpha0 nan nan nan nan nan nan nan nan]; 
        alpha2 = [alpha0 nan nan nan nan nan nan nan nan]; 

        % Decision noise, and side bias for this game depend on 
        % the horizon. Decision noise additionally depends on
        % information condition
        sigma_g = decision_noises(mdp.C1(g));
        
        num_choices = sum(~isnan(actions(g,:)));

        for t=1:num_choices  % loop over forced-choice trials
            if t >= 5
                % compute UCB
                if mdp.C1(g)==1
                    T = 1;
                else
                    T = 1+params.info_bonus;
                end
                UCB1 = mu1(t) + (2*log(T)/(sum(actions(1,1:t-1) == 1)*params.outcome_informativeness))^.5 + bias;
                UCB2 = mu2(t) + (2*log(T)/(sum(actions(1,1:t-1) == 2)*params.outcome_informativeness))^.5;

                % total uncertainty = add variance of both arms and then square root 
                % total uncertainty
                Y = 1;
                Y = 1+params.RE;
                
                % decision_noise = total uncertainty + 2*log(Y)
                
                % probability of choosing bandit 1
                p = exp(UCB1 / sigma_g) / (exp(UCB1 / sigma_g) + exp(UCB2 / sigma_g));
                action_probs(g,t) = mod(actions(g,t),2)*p + (1-mod(actions(g,t),2))*(1-p);
            end
            
            % left bandit choice so mu1 updates
            if (actions(g,t) == 1) 
                % update LR
                alpha1(t+1) = 1/( 1/(alpha1(t) + alpha_d) + 1 );
                alpha2(t+1) = 1/( 1/(alpha2(t) + alpha_d) );
                exp_vals(g,t) = mu1(t);
                pred_errors(g,t) = (rewards(g,t) - exp_vals(g,t));
                alpha(g,t) = alpha1(t+1);
                pred_errors_alpha(g,t) = alpha1(t+1) * pred_errors(g,t); % confirm that alpha here should be t+1
                mu1(t+1) = mu1(t) + pred_errors_alpha(g,t);
                mu2(t+1) = mu2(t); 
            else % right bandit choice so mu2 updates
                % update LR
                alpha1(t+1) = 1/( 1/(alpha1(t) + alpha_d) ); 
                alpha2(t+1) = 1/( 1/(alpha2(t) + alpha_d) + 1 );
                exp_vals(g,t) = mu2(t);
                pred_errors(g,t) = (rewards(g,t) - exp_vals(g,t));
                alpha(g,t) = alpha2(t+1);
                pred_errors_alpha(g,t) = alpha2(t+1) * pred_errors(g,t);
                mu1(t+1) = mu1(t);
                mu2(t+1) = mu2(t) + pred_errors_alpha(g,t);
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
        alpha1 = [alpha0 nan nan nan nan nan nan nan nan]; 
        alpha2 = [alpha0 nan nan nan nan nan nan nan nan]; 

        % Decision noise, and side bias for this game depend on 
        % the horizon. Decision noise additionally depends on
        % information condition
        sigma_g = decision_noises(mdp.C1(g));
        
        num_choices = sum(~isnan(actions(g,:)));

        for t=1:num_choices  % loop over forced-choice trials
            simmed_free_choices(g,t) = actions(g,t);
            if t >= 5
                % compute UCB
                if mdp.C1(g)==1
                    T = 1;
                else
                    T = 1+params.info_bonus;
                end
                UCB1 = mu1(t) + (2*log(T)/(sum(actions(1,1:t-1) == 1)*params.outcome_informativeness))^.5 + bias;
                UCB2 = mu2(t) + (2*log(T)/(sum(actions(1,1:t-1) == 2)*params.outcome_informativeness))^.5;

                % probability of choosing bandit 1 (left)
                p = exp(UCB1 / sigma_g) / (exp(UCB1 / sigma_g) + exp(UCB2 / sigma_g));
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
            % left bandit choice so mu1 updates
            if (simmed_free_choices(g,t) == 1) 
                % update LR
                alpha1(t+1) = 1/( 1/(alpha1(t) + alpha_d) + 1 );
                alpha2(t+1) = 1/( 1/(alpha2(t) + alpha_d) );
                sim_exp_vals(g,t) = mu1(t);
                sim_pred_errors(g,t) = (simmed_rewards(g,t) - sim_exp_vals(g,t));
                sim_alpha(g,t) = alpha1(t+1);
                sim_pred_errors_alpha(g,t) = alpha1(t+1) * sim_pred_errors(g,t); % confirm that alpha here should be t+1
                mu1(t+1) = mu1(t) + sim_pred_errors_alpha(g,t);
                mu2(t+1) = mu2(t); 
            else % right bandit choice so mu2 updates
                % update LR
                alpha1(t+1) = 1/( 1/(alpha1(t) + alpha_d) ); 
                alpha2(t+1) = 1/( 1/(alpha2(t) + alpha_d) + 1 );
                sim_exp_vals(g,t) = mu2(t);
                sim_pred_errors(g,t) = (simmed_rewards(g,t) - sim_exp_vals(g,t));
                sim_alpha(g,t) = alpha2(t+1);
                sim_pred_errors_alpha(g,t) = alpha2(t+1) * sim_pred_errors(g,t);
                mu1(t+1) = mu1(t);
                mu2(t+1) = mu2(t) + sim_pred_errors_alpha(g,t);
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