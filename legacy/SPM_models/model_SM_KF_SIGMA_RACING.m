function model_output = model_SM_KF_SIGMA_RACING(params, actions_and_rts, rewards, mdp, sim)
    % note that mu2 == right bandit ==  actions2
    num_games = mdp.num_games; % num of games
    num_choices_to_fit = mdp.settings.num_choices_to_fit;
    num_forced_choices = mdp.num_forced_choices;
    num_free_choices_big_hor = mdp.num_free_choices_big_hor;
    num_choices_big_hor = num_forced_choices + num_free_choices_big_hor;
    max_rt = mdp.settings.max_rt;
    
    % initialize params
    initial_sigma = params.initial_sigma;
    initial_mu = params.initial_mu;
    reward_sensitivity = params.reward_sensitivity;   

    sigma_d = params.sigma_d;
    side_bias = params.side_bias;
    sigma_r = params.sigma_r;
    cong_base_info_bonus = params.cong_base_info_bonus;
    incong_base_info_bonus = params.incong_base_info_bonus;
    cong_directed_exp = params.cong_directed_exp;
    incong_directed_exp = params.incong_directed_exp;
    random_exp = params.random_exp;
    baseline_noise = params.baseline_noise;
    decision_thresh_baseline = params.decision_thresh_baseline;
    wd = params.wd;
    ws = params.ws;
    V0 = params.V0;
    
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
        
        num_choices_in_this_game = sum(~isnan(rewards(g,:))); 
        % Get the number of choices to loop over depending on how many free
        % choices we're fititng and the number of choices in this game
        num_choices_to_loop_over = min(num_choices_in_this_game, num_choices_to_fit + num_forced_choices);

        for t=1:num_choices_to_loop_over  % loop over forced-choice trials
            if t >= 5
                num_trials_left = num_choices_to_loop_over - t + 1;
                reward_diff = mu2(t) - mu1(t);
                % relative uncertainty is the difference in uncertainty
                relative_uncert = sigma2(g,t) - sigma1(g,t);
                % total uncertainty is variance of both arms
                total_uncert = (sigma1(g,t)^2 + sigma2(g,t)^2)^.5;
                % If both reward difference and UCB difference push in same direction, use the cong tradeoff; otherwise, use incong.
                 if (reward_diff*relative_uncert >= 0)
                    rel_uncert_scaler = (exp(num_trials_left-1)-1)*cong_directed_exp+ cong_base_info_bonus;
                 else
                    rel_uncert_scaler = (exp(num_trials_left-1)-1)*incong_directed_exp+ incong_base_info_bonus;
                 end

                 % SET DDM Parameters
                 drift = (rel_uncert_scaler*relative_uncert) + (reward_diff/log(1 + exp(min(baseline_noise + total_uncert*(num_trials_left-1)*random_exp,700))));
                 %starting_bias_untransformed = (side_bias + rdiff_bias_mod*reward_diff/total_uncert);
                 % Transform starting_bias to be between 0 and 1 using sigmoid
                 %starting_bias = 1 / (1 + exp(-starting_bias_untransformed));
                 decision_thresh(g,t) = decision_thresh_baseline;
                
                
                %BUILD 2‑ACCUMULATOR RACE%
                % drifts: Based on Advantaged Racing Diffusion 
                v1 = V0 + wd*(mu1(t) - mu2(t)) + ws*(mu1(t) + mu2(t));
                v2 = V0 + wd*(mu2(t) - mu1(t)) +  ws*(mu1(t) + mu2(t));
                v_cell = {v1, v2};
                % across‐trial noise scaling variability. Just dummy values
                % for now
                s_cell = {1, 1};
      
                B1     = decision_thresh(g,t); % Threshold for option 1
                B2     = B1;                   % Threshold for option 2
                B_cell = { B1, B2};
                % no trial‐to‐trial threshold variability for now
                A_cell =  {0, 0};
                t0_i   = {0,0}; % No non-decision time for now
    
                %SIMULATION OR FITTING
                if sim
                    % call the race model for 1 trial
                    % Simulate a choice/RT based on random sampling
                    if mdp.num_samples_to_draw_from_pdf > 0
                        out = rWaldRace(1, ...      % simulate 1 trial
                                        v_cell, ... % 1×2 cell of drifts
                                        B_cell, ... % 1×2 cell of thresholds
                                        A_cell, ... % 1×2 cell of threshold variabilities
                                        t0_i,   ... % scalar non‑decision time
                                        s_cell);    % 1×2 cell of noise scales
                    
                        % extract simulated RT and choice
                        simmed_rt   = out.RT;
                        simmed_choice = out.R;        % 1 or 2
                    else
                        % Simulate a choice/RT based on the maximum of the pdf
                        % Get the max pdf for the left choice
                        fun = @(rt) dWaldRace(rt, 1, A_cell, B_cell, t0_i, v_cell, s_cell );
                        % This function gets min so we use a negative fun
                        [rt_max_left, neg_max_pdf_left] = fminbnd(@(rt) -fun(rt), 0, max_rt); 
                        % Get the max pdf of the right choice
                        fun = @(rt) dWaldRace(rt, 2, A_cell, B_cell, t0_i, v_cell, s_cell );
                        % This function gets min so we use a negative fun
                        [rt_max_right, neg_max_pdf_right] = fminbnd(@(rt) -fun(rt), 0, max_rt); 
                        % Take the choice/RT associated with the biggest pdf
                        % (remember it's negative)
                        if neg_max_pdf_left < neg_max_pdf_right
                            simmed_rt = rt_max_left;
                            simmed_choice = 1;
                        else
                            simmed_rt = rt_max_right;
                            simmed_choice = 2;
                        end

                    end
                    actions(g,t) = simmed_choice;
                    if simmed_choice == 2
                        rewards(g,t) = mdp.bandit2_schedule(g,t);
                    else
                        rewards(g,t) = mdp.bandit1_schedule(g,t);
                    end
                    rts(g,t) = simmed_rt;
                end
    
                % only compute likelihood if RT is valid
                if rts(g,t)>0 && rts(g,t)<max_rt               
                    % DEFECATING PDF @ observed RT
                    rt_pdf(g,t) = dWaldRace( ...
                        rts(g,t), ...   % scalar RT
                        actions(g,t), ...% action actually taken
                        A_cell, ...     % threshold variability
                        B_cell, ...     % thresholds
                        t0_i, ...       % non‐decision time
                        v_cell, ...     % drift rates
                        s_cell );       % noise scaling variability
    
                    % CHOICE PROBABILITY UP TO max_rt %
                    resp = actions(g,t);   % 1 or 2 (or up to m)
                    fun = @(y) dWaldRace(y, resp, A_cell, B_cell, t0_i, v_cell, s_cell );  
                    action_probs(g,t) = integral(fun, 0, max_rt, 'ArrayValued', true);
               
                    model_acc(g,t) = (action_probs(g,t) > 0.5);
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