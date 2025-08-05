function model_output = model_SM_KF_SIGMA_logistic_DDM(params, actions_and_rts, rewards, mdp, sim)
    % note that mu2 == right bandit ==  actions2
    num_games = mdp.num_games; % num of games
    num_choices_to_fit = mdp.settings.num_choices_to_fit;
    num_forced_choices = mdp.num_forced_choices;
    num_free_choices_big_hor = mdp.num_free_choices_big_hor;
    num_choices_big_hor = num_forced_choices + num_free_choices_big_hor;
    max_rt = mdp.settings.max_rt;
    
    % initialize params
    sigma_d = params.sigma_d;
    side_bias_small_hor = params.side_bias_small_hor;
    side_bias_big_hor = params.side_bias_big_hor;
    sigma_r = params.sigma_r;
    initial_sigma = params.initial_sigma;
    initial_mu = params.initial_mu;
    reward_sensitivity = params.reward_sensitivity;   
    info_bonus_small_hor = params.info_bonus_small_hor;
    info_bonus_big_hor = params.info_bonus_big_hor;
    dec_noise_small_hor = params.dec_noise_small_hor;
    dec_noise_big_hor = params.dec_noise_big_hor;
    
   
    
    % initialize variables

    actions = actions_and_rts.actions;
    rts = actions_and_rts.RTs;
    rt_pdf = nan(num_games,num_choices_big_hor);
    action_probs = nan(num_games,num_choices_big_hor);
    model_acc = nan(num_games,num_choices_big_hor);
    
    pred_errors = nan(num_games,num_choices_big_hor+1);
    pred_errors_alpha = nan(num_games,num_choices_big_hor+1);
    exp_vals = nan(num_games,num_choices_big_hor+1);
    alpha = nan(num_games,num_choices_big_hor+1);
    sigma1 = [initial_sigma * ones(num_games,1), zeros(num_games,num_choices_big_hor-1)];
    sigma2 = [initial_sigma * ones(num_games,1), zeros(num_games,num_choices_big_hor-1)];
    total_uncertainty = nan(num_games,num_choices_big_hor);
    estimated_mean_diff = nan(num_games,num_choices_big_hor);
    relative_uncertainty_of_choice = nan(num_games,num_choices_big_hor);
    change_in_uncertainty_after_choice = nan(num_games,num_choices_big_hor);

    num_invalid_rts = 0;

    decision_thresh = nan(num_games,num_choices_big_hor);

    
    for g=1:num_games  % loop over games
        % values
        mu1 = [initial_mu nan nan nan nan nan nan nan nan];
        mu2 = [initial_mu nan nan nan nan nan nan nan nan];

        % learning rates 
        alpha1 = nan(1,num_games); 
        alpha2 = nan(1,num_games); 
        
        % if small horizon Game, use small horizon info bonus, decision noise, and side bias
        if mdp.horizon_type(g) == 1
            info_bonus = info_bonus_small_hor;
            decision_noise = dec_noise_small_hor;
            side_bias = side_bias_small_hor;
        else
        % if big horizon Game, use big horizon info bonus, decision noise, and side bias
            info_bonus = info_bonus_big_hor;
            decision_noise = dec_noise_big_hor;
            side_bias = side_bias_big_hor;
        end

        for t=1:num_forced_choices+1  % loop over forced choices and first free choice
            if t == num_forced_choices+1
                % Get the reward difference
                reward_diff = mu2(t) - mu1(t);
                % Get the information difference (+1 when fewer options are
                % shown on the right, -1 when fewer options are shown on
                % left)
                info_diff = mdp.forced_choice_info_diff(g);
                % total uncertainty is variance of both arms
                % probability of choosing bandit 1
                p = 1 / (1 + exp((reward_diff+(info_diff*info_bonus)+side_bias)/(decision_noise)));
                            

                % Set DDM params
                % DRIFT
                % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                drift = params.drift_baseline;
                if any(contains(mdp.settings.drift_mapping, 'reward_diff'))
                    drift = drift + params.drift_reward_diff_mod*reward_diff;
                end
                if any(contains(mdp.settings.drift_mapping, 'info_diff'))
                    drift = drift + info_diff*info_bonus;
                end
                if any(contains(mdp.settings.drift_mapping, 'side_bias'))
                    drift = drift + side_bias;
                end    
                if any(contains(mdp.settings.drift_mapping, 'decision_noise'))
                    drift = drift/decision_noise;
                end 
                    
                % STARTING BIAS
                % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Transform baseline starting bias so in sigmoid space
                % (must be between 0 and 1)
                starting_bias = log(params.starting_bias_baseline/(1-params.starting_bias_baseline));
                if any(contains(mdp.settings.bias_mapping, 'reward_diff'))
                    starting_bias = starting_bias + params.starting_bias_reward_diff_mod*reward_diff;
                end
                if any(contains(mdp.settings.bias_mapping, 'info_diff'))
                    starting_bias = starting_bias + info_diff*info_bonus;
                end
                if any(contains(mdp.settings.bias_mapping, 'side_bias'))
                    starting_bias = starting_bias + side_bias;
                end    
                % Transform starting_bias to be between 0 and 1 using sigmoid
                starting_bias = 1 / (1 + exp(-starting_bias));
                
                % DECISION THRESHOLD
                % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % bounded to be less than 500
                decision_thresh(g,t) = params.decision_thresh_baseline;
                if any(contains(mdp.settings.thresh_mapping, 'decision_noise'))
                    decision_thresh(g,t) = decision_noise;
                end
                
                if sim
                    % higher drift rate / bias entails greater prob of
                    % choosing bandit 2

                    % Simulate a choice/RT based on random sampling
                    if mdp.num_samples_to_draw_from_pdf > 0
                        [simmed_rt, chose_right] = simulate_DDM(drift, decision_thresh(g,t), 0, starting_bias, 1, .001, realmax);
                    else
                        % Simulate a choice/RT based on the maximum of the pdf
                        % Get the max pdf for the left choice
                        fun = @(rt) wfpt(rt, drift, decision_thresh(g,t), starting_bias);
                        % This function gets min so we use a negative fun
                        [rt_max_left, neg_max_pdf_left] = fminbnd(@(rt) -fun(rt), 0, max_rt); 
                        % Get the max pdf of the right choice (remember we
                        % have to invert drift and starting bias because
                        % wfpt gives the pdf for bottom boundary and we've
                        % decided bottom boundary is left choice
                        fun = @(rt) wfpt(rt, -drift, decision_thresh(g,t), 1-starting_bias);
                        % This function gets min so we use a negative fun
                        [rt_max_right, neg_max_pdf_right] = fminbnd(@(rt) -fun(rt), 0, max_rt); 
                        % Take the choice/RT associated with the biggest pdf
                        % (remember it's negative)
                        if neg_max_pdf_left < neg_max_pdf_right
                            simmed_rt = rt_max_left;
                            chose_right = 0;
                        else
                            simmed_rt = rt_max_right;
                            chose_right = 1;
                        end
                    end
                    if chose_right
                        actions(g,t) = 2;
                        rewards(g,t) = mdp.bandit2_schedule(g,t);
                    else
                        actions(g,t) = 1;
                        rewards(g,t) = mdp.bandit1_schedule(g,t);
                    end
                    rts(g,t) = simmed_rt;
                end
                % if RT is less than max and greater than 0, consider in log likelihood
                if rts(g,t) < max_rt && rts(g,t) > 0
                    if  actions(g,t) == 2 % invert the drift rate if they chose right since the bottom boundary corresponds to left. 
                        drift = drift * -1;
                        starting_bias = 1 - starting_bias;
                    end
                    % We've decided that the bottom boundary will
                    % correspond to the left choice, so 
                    % negative drift and lower bias entail greater
                    % probability of choosing left bandit
                    rt_pdf(g,t) = wfpt(rts(g,t), drift, decision_thresh(g,t), starting_bias);
                    % plot_ddm_pdf(drift,starting_bias,decision_thresh);
                    action_probs(g,t) = integral(@(y) wfpt(y,drift,decision_thresh(g,t),starting_bias),0,max_rt);
                    model_acc(g,t) =  action_probs(g,t) > .5;
               end
                
                
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
            % save total uncertainty and reward difference
            total_uncertainty(g,t) = ((sigma1(g,t)^2)+(sigma2(g,t)^2))^.5;
            estimated_mean_diff(g,t) = mu2(t) - mu1(t);

            if ~sim
                if rts(g,t) >= max_rt || rts(g,t) <= 0
                    num_invalid_rts = num_invalid_rts + 1;
                end
            end

        end
    end

    
    model_output.action_probs = action_probs;
    model_output.rt_pdf = rt_pdf;
    model_output.model_acc = model_acc;
    model_output.num_invalid_rts = num_invalid_rts;
    
    model_output.exp_vals = exp_vals;
    model_output.pred_errors = pred_errors;
    model_output.pred_errors_alpha = pred_errors_alpha;
    model_output.alpha = alpha;
    model_output.sigma1 = sigma1;
    model_output.sigma2 = sigma2;
    model_output.actions = actions;
    model_output.rewards = rewards;
    model_output.rts = rts;
    model_output.relative_uncertainty_of_choice = relative_uncertainty_of_choice;
    model_output.total_uncertainty = total_uncertainty;
    model_output.estimated_mean_diff = estimated_mean_diff;
    model_output.change_in_uncertainty_after_choice = change_in_uncertainty_after_choice;

end