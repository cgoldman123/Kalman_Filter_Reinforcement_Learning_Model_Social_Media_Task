function model_output = model_SM_KF_DDM_all_choices(params, actions_and_rts, rewards, mdp, sim)
    dbstop if error;
    % note that mu2 == right bandit ==  c=2 == free choice = 1
    G = mdp.G; % num of games
    T = 9; % num of choices
    
    max_rt = mdp.settings.max_rt;
    
    % initialize params
    sigma_d = params.sigma_d;
    side_bias = params.side_bias;
    sigma_r = params.sigma_r;
    initial_sigma = params.initial_sigma;
    h1_info_bonus = params.h1_info_bonus;
    h1_dec_noise = params.h1_dec_noise;
    initial_mu = params.initial_mu;
    reward_sensitivity = params.reward_sensitivity;    
    % indicate if want one parameter to control DE/RE or keep separate
    if any(strcmp('DE_RE_horizon', fieldnames(params)))
        DE_RE_horizon = params.DE_RE_horizon;
    else
        h5_baseline_info_bonus = params.h5_baseline_info_bonus;
        h5_slope_info_bonus = params.h5_slope_info_bonus;
        h5_baseline_dec_noise = params.h5_baseline_dec_noise;
        h5_slope_dec_noise = params.h5_slope_dec_noise;
    end
    
    % initialize variables
    if sim
        actions = nan(G,T);
        rts = nan(G,T);
    else 
        actions = actions_and_rts.actions;
        rts = actions_and_rts.RTs;
        rt_pdf = nan(G,9);
        action_probs = nan(G,9);
        model_acc = nan(G,9);
    end
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
        
        num_choices = sum(~isnan(rewards(g,:))); 

        for t=1:num_choices  % loop over forced-choice trials
            if t >= 5
                % total uncertainty is variance of both arms
                total_uncertainty = (sigma1(t)^2 + sigma2(t)^2)^.5;
                if mdp.C1(g)==1 % horizon is 1
                    UCB_diff = h1_info_bonus* (sigma1(t) - sigma2(t));
                    decision_noise = h1_dec_noise*total_uncertainty;
                else   % horizon is 5
                    decision_noise = h5_baseline_dec_noise + h5_slope_dec_noise*(t-5)*total_uncertainty;
                    UCB_diff = (h5_baseline_info_bonus+(h5_slope_info_bonus*(t-5)))*(sigma1(t) - sigma2(t));
                end
                reward_diff = mu1(t) - mu2(t);
       
                % probability of choosing bandit 1
                p = 1 / (1 + exp(-(reward_diff+UCB_diff+side_bias)/(decision_noise)));
                
                % Set DDM params
                % DRIFT
                % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if isempty(mdp.settings.drift_mapping)
                    drift = params.drift;
                else
                    drift = params.drift_baseline;
                    if contains(mdp.settings.drift_mapping, 'action_prob')
                        drift = drift + params.drift_action_prob_mod*(p - .5);
                    end
                    if contains(mdp.settings.drift_mapping, 'reward_diff')
                        drift = drift + params.drift_reward_diff_mod*reward_diff;
                    end
                    if contains(mdp.settings.drift_mapping, 'UCB_diff')
                        drift = drift + params.drift_UCB_diff_mod*UCB_diff;
                    end
                    if contains(mdp.settings.drift_mapping, 'side_bias')
                        drift = drift + side_bias;
                    end    
                end
                    
                % STARTING BIAS
                % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if isempty(mdp.settings.bias_mapping)
                    starting_bias = params.starting_bias;
                else
                    starting_bias = .5;
                    if contains(mdp.settings.bias_mapping, 'action_prob')
                        starting_bias = starting_bias + params.bias_action_prob_mod*(p - .5);
                    end
                    if contains(mdp.settings.bias_mapping, 'reward_diff')
                        starting_bias = starting_bias + params.bias_reward_diff_mod*reward_diff;
                    end
                    if contains(mdp.settings.bias_mapping, 'UCB_diff')
                        starting_bias = starting_bias + params.bias_UCB_diff_mod*UCB_diff;
                    end
                    if contains(mdp.settings.bias_mapping, 'side_bias')
                        starting_bias = starting_bias + side_bias;
                    end    
                end                    
                    
                % DECISION THRESHOLD
                % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if isempty(mdp.settings.thresh_mapping)
                    decision_thresh = params.decision_thresh;
                else
                    decision_thresh = params.thresh_baseline;
                    if contains(mdp.settings.thresh_mapping, 'action_prob')
                        decision_thresh = decision_thresh + params.thresh_action_prob_mod*(p - .5);
                    end
                    if contains(mdp.settings.thresh_mapping, 'reward_diff')
                        decision_thresh = decision_thresh + params.thresh_reward_diff_mod*reward_diff;
                    end
                    if contains(mdp.settings.thresh_mapping, 'UCB_diff')
                        decision_thresh = decision_thresh + params.thresh_UCB_diff_mod*UCB_diff;
                    end
                    if contains(mdp.settings.thresh_mapping, 'side_bias')
                        decision_thresh = decision_thresh + side_bias;
                    end 
                    decision_thresh = log(1+exp(decision_thresh));

                end   
                
                if sim
                    % higher drift rate / bias entails greater prob of
                    % choosing bandit 1
                    [simmed_rt, chose_left] = simulate_DDM(drift, decision_thresh, 0, starting_bias, 1, .001, realmax);
                    % accepted dot motion
                    if chose_left
                        actions(g,t) = 1;
                        rewards(g,t) = mdp.bandit1_schedule(g,t);
                    else
                        actions(g,t) = 2;
                        rewards(g,t) = mdp.bandit2_schedule(g,t);
                    end
                    rts(g,t) = simmed_rt;
                else
                    % if RT is less than max, consider in log likelihood
                    if rts(g,t) < max_rt
                        if  actions(g,t) == 1 % chose left
                            % negative drift and lower bias entail greater
                            % probability of choosing left bandit
                            drift = drift * -1;
                            starting_bias = 1 - starting_bias;
                        end
                        rt_pdf(g,t) = wfpt(rts(g,t), drift, decision_thresh, starting_bias);
                        action_probs(g,t) = integral(@(y) wfpt(y,drift,decision_thresh,starting_bias),0,max_rt);
                        model_acc(g,t) =  action_probs(g,t) > .5;
                   end
                end
                
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

    
    if ~sim
        model_output.action_probs = action_probs;
        model_output.rt_pdf = rt_pdf;
        model_output.model_acc = model_acc;
    end
    model_output.exp_vals = exp_vals;
    model_output.pred_errors = pred_errors;
    model_output.pred_errors_alpha = pred_errors_alpha;
    model_output.alpha = alpha;
    model_output.actions = actions;
    model_output.rewards = rewards;
    model_output.rts = rts;

end