function model_output = model_SM_obs_means_logistic(params, actions_and_rts, rewards, mdp, sim)
    dbstop if error;
    % note that mu2 == right bandit ==  c=2 == free choice = 1   
    num_games = mdp.processed_data.num_games; % num of games
    num_choices_to_fit = mdp.num_choices_to_fit;
    num_forced_choices = mdp.processed_data.num_forced_choices;
    num_free_choices_big_hor = mdp.processed_data.num_free_choices_big_hor;
    num_choices_big_hor = num_forced_choices + num_free_choices_big_hor;
    
    % initialize params
    %sigma_d = params.sigma_d;
    side_bias_small_hor = params.side_bias_small_hor;
    side_bias_big_hor = params.side_bias_big_hor;
    %sigma_r = params.sigma_r;
    %initial_sigma = params.initial_sigma;
    %initial_mu = params.initial_mu;
    %reward_sensitivity = params.reward_sensitivity;   
    info_bonus_small_hor = params.info_bonus_small_hor;
    info_bonus_big_hor = params.info_bonus_big_hor;
    dec_noise_small_hor = params.dec_noise_small_hor;
    dec_noise_big_hor = params.dec_noise_big_hor;
    
   
    
    % initialize variables
    actions = actions_and_rts.actions;
    action_probs = nan(num_games,num_choices_big_hor);
    model_acc = nan(num_games,num_choices_big_hor);
    rts = nan(num_games,num_choices_big_hor);

    pred_errors = nan(num_games,num_choices_big_hor+1);
    pred_errors_alpha = nan(num_games,num_choices_big_hor+1);
    exp_vals = nan(num_games,num_choices_big_hor+1);
    alpha = nan(num_games,10);
    sigma1 = nan(num_games,9);
    sigma2 = nan(num_games,9);
    total_uncertainty = nan(num_games,num_choices_big_hor);
    relative_uncertainty_of_choice = nan(num_games,num_choices_big_hor);
    change_in_uncertainty_after_choice = nan(num_games,num_choices_big_hor);
    estimated_mean_diff = nan(num_games,num_choices_big_hor);
    


    
    for g=1:num_games  % loop over games
        % values
        mu1 = nan(1,5);
        mu2 = nan(1,5);
        
        % if H1 Game, use H1 info bonus, decision noise, and side bias
        if mdp.processed_data.horizon_type(g) == 1
            info_bonus = info_bonus_small_hor;
            decision_noise = dec_noise_small_hor;
            side_bias = side_bias_small_hor;
        else
        % if H5 Game, use H5 info bonus, decision noise, and side bias
            info_bonus = info_bonus_big_hor;
            decision_noise = dec_noise_big_hor;
            side_bias = side_bias_big_hor;
        end


        mu1(5) = mean(mdp.processed_data.bandit1_schedule(g,1:4));
        mu2(5) = mean(mdp.processed_data.bandit2_schedule(g,1:4));

        
        for t=1:5  % loop over 4 forced choices and 1 free choice trials
            if t == 5
                % Get the reward difference
                reward_diff = mu2(t) - mu1(t);
                % Get the information difference (+1 when fewer options are
                % shown on the right, -1 when fewer options are shown on
                % left)
                info_diff = mdp.processed_data.forced_choice_info_diff(g);
                % total uncertainty is variance of both arms
                % probability of choosing bandit 1
                p = 1 / (1 + exp((reward_diff+(info_diff*info_bonus)+side_bias)/(decision_noise)));
                            
                %simulate behavior
                if sim
                    u = rand(1,1);
                    if u <= p
                        actions(g,t) = 1;
                        rewards(g,t) = mdp.processed_data.bandit1_schedule(g,t);
                    else
                        actions(g,t) = 2;
                        rewards(g,t) = mdp.processed_data.bandit2_schedule(g,t);
                    end
                end
                action_probs(g,t) = mod(actions(g,t),2)*p + (1-mod(actions(g,t),2))*(1-p);
                model_acc(g,t) =  action_probs(g,t) > .5;  
                
            end
          
            % Store total uncertainty unless someone wants it later
            % total_uncertainty(g,t) = (sigma1(g,t)^2 + sigma2(g,t)^2)^.5;

            % left bandit choice so mu1 updates
            % if (actions(g,t) == 1) 
            %     % save relative uncertainty of choice
            %     % relative_uncertainty_of_choice(g,t) = sigma1(g,t) - sigma2(g,t);
            % 
            %     % update sigma and LR
            %     % temp = 1/(sigma1(g,t)^2 + sigma_d^2) + 1/(sigma_r^2);
            %     % sigma1(g,t+1) = (1/temp)^.5;
            %     % change_in_uncertainty_after_choice(g,t) = sigma1(g,t+1) - sigma1(g,t);
            %     % alpha1(t) = (sigma1(g,t+1)/(sigma_r))^2;
            % 
            %     % temp = sigma2(g,t)^2 + sigma_d^2;
            %     % sigma2(g,t+1) = temp^.5;
            % 
            %     exp_vals(g,t) = mu1(t);
            %     pred_errors(g,t) = (reward_sensitivity*rewards(g,t)) - exp_vals(g,t);
            %     alpha(g,t) = alpha1(t);
            %     pred_errors_alpha(g,t) = alpha1(t) * pred_errors(g,t);
            %     mu1(t+1) = mu1(t) + pred_errors_alpha(g,t);
            %     mu2(t+1) = mu2(t); 
            % else % right bandit choice so mu2 updates
            %     % save relative uncertainty of choice
            %     relative_uncertainty_of_choice(g,t) = sigma2(g,t) - sigma1(g,t);
            %     % update LR
            %     temp = 1/(sigma2(g,t)^2 + sigma_d^2) + 1/(sigma_r^2);
            %     sigma2(g,t+1) = (1/temp)^.5;
            %     change_in_uncertainty_after_choice(g,t) = sigma2(g,t+1) - sigma2(g,t);
            %     alpha2(t) = (sigma2(g,t+1)/(sigma_r))^2;
            % 
            %     temp = sigma1(g,t)^2 + sigma_d^2;
            %     sigma1(g,t+1) = temp^.5;
            % 
            %     exp_vals(g,t) = mu2(t);
            %     pred_errors(g,t) = (reward_sensitivity*rewards(g,t)) - exp_vals(g,t);
            %     alpha(g,t) = alpha2(t);
            %     pred_errors_alpha(g,t) = alpha2(t) * pred_errors(g,t);
            %     mu2(t+1) = mu2(t) + pred_errors_alpha(g,t);
            %     mu1(t+1) = mu1(t);
            % end

            % save reward difference
            estimated_mean_diff(g,t) = mu2(t) - mu1(t);

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
    model_output.estimated_mean_diff = estimated_mean_diff;
    model_output.change_in_uncertainty_after_choice = change_in_uncertainty_after_choice;
    model_output.rts = rts;


end