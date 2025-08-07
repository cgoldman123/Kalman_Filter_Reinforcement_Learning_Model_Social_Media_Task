    
function varargout = transform_params_SM(trans_or_untrans, params_struct,params_field)
    if isempty(params_field)
        params = [];
        prior_sigma = [];
    else
        if strcmp(trans_or_untrans,"transform")
            for i = 1:length(params_field)
                field = params_field{i};
                % transform the parameters that we fit
                if ismember(field, {'learning_rate', 'learning_rate_pos', 'learning_rate_neg', 'noise_learning_rate', 'alpha_start', 'alpha_inf', 'associability_weight', ...
                        'starting_bias_baseline'})
                    params.(field) = log(params_struct.(field)/(1-params_struct.(field)));  % bound between 0 and 1
                    prior_sigma.(field) = 1/4;
                elseif ismember(field, {'h1_dec_noise', 'h5_baseline_dec_noise', 'h5_sloparams_dec_noise', ...
                        'initial_sigma', 'initial_sigma_r', 'initial_mu', 'initial_associability', ...
                        'drift_action_prob_mod', 'drift_reward_diff_mod', 'drift_UCB_diff_mod',...
                        'starting_bias_action_prob_mod', 'starting_bias_reward_diff_mod', 'starting_bias_UCB_diff_mod',...
                        'decision_thresh_action_prob_mod', 'decision_thresh_reward_diff_mod', 'decision_thresh_UCB_diff_mod', 'decision_thresh_decision_noise_mod' ...
                        'outcome_informativeness', ...
                        'reward_sensitivity', 'DE_RE_horizon'})
                    params.(field) = log(params_struct.(field));               % in log-space (to keep positive)
                    prior_sigma.(field) = 1/4;
                elseif ismember(field, {'h5_baseline_info_bonus', 'h5_sloparams_info_bonus', 'h1_info_bonus', ...
                        'side_bias', 'side_bias_h1', 'side_bias_h5', 'info_bonus', ...
                        'drift_baseline', 'drift'})
                    params.(field) = params_struct.(field); 
                    prior_sigma.(field) = 1/4;
                elseif any(strcmp(field,{'nondecision_time'})) % bound between .1 and .3
                    params.(field) =  -log((0.3 - 0.1) ./ (params_struct.(field) - 0.1) - 1);  
                    prior_sigma.(field) = 1/4;
                elseif any(strcmp(field,{'decision_thresh_baseline'})) % bound greater than .5 and less than 1000
                    params.(field) =  -log((1000 - .5) ./ (params_struct.(field) - .5) - 1); 
                    prior_sigma.(field) = 1/4;
                elseif any(strcmp(field,{'sigma_d'})) % bound between 0 and 40
                    params.(field) =  -log((40) ./ (params_struct.(field)) - 1);  
                    prior_sigma.(field) = 1/4;
                elseif any(strcmp(field,{'sigma_r'})) % bound between 0 and 40
                    params.(field) =  -log((40) ./ (params_struct.(field)) - 1);   
                    prior_sigma.(field) = 1/4;
                elseif any(strcmp(field,{'baseline_info_bonus'})) % bound between 0 and 40
                    params.(field) = params_struct.(field); 
                    prior_sigma.(field) = 1/4;
                elseif any(strcmp(field,{'directed_exp'})) 
                    params.(field) = params_struct.(field); 
                    prior_sigma.(field) = 1/4;
                elseif any(strcmp(field,{'baseline_noise'})) 
                    params.(field) = log(params_struct.(field));               % in log-space (to keep positive)
                    prior_sigma.(field) = 1/4;
                elseif any(strcmp(field,{'random_exp'})) % bound between 0 and 40
                    params.(field) = log(params_struct.(field));               % in log-space (to keep positive)
                    prior_sigma.(field) = 1/4;
                else   
                    disp(field);
                    error("Param not proparamsrly transformed");
                end
            end
        elseif strcmp(trans_or_untrans,"untransform")
            prior_sigma = [];
            for i = 1:length(params_field)
                field = params_field{i};
                if ismember(field,{'learning_rate', 'learning_rate_pos', 'learning_rate_neg', 'noise_learning_rate', 'alpha_start', 'alpha_inf', 'associability_weight', ...
                        'starting_bias_baseline'})
                    params.(field) = 1/(1+exp(-params_struct.(field)));
                elseif ismember(field,{'h1_dec_noise', 'h5_baseline_dec_noise', 'h5_slope_dec_noise', ...
                        'initial_sigma', 'initial_sigma_r', 'initial_mu', 'initial_associability', ...
                        'drift_action_prob_mod', 'drift_reward_diff_mod', 'drift_UCB_diff_mod',...
                        'starting_bias_action_prob_mod', 'starting_bias_reward_diff_mod', 'starting_bias_UCB_diff_mod',...
                        'decision_thresh_action_prob_mod', 'decision_thresh_reward_diff_mod', 'decision_thresh_UCB_diff_mod', 'decision_thresh_decision_noise_mod'...
                        'outcome_informativeness',  'random_exp', 'baseline_noise',...
                        'reward_sensitivity', 'DE_RE_horizon'})
                    params.(field) = exp(params_struct.(field));
                elseif ismember(field,{'h5_baseline_info_bonus', 'h5_slope_info_bonus', 'h1_info_bonus', 'baseline_info_bonus',...
                        'side_bias', 'side_bias_h1', 'side_bias_h5', 'info_bonus',...
                        'drift_baseline', 'drift', 'directed_exp'})
                    params.(field) = params_struct.(field);
                elseif any(strcmp(field,{'nondecision_time'}))
                    params.(field) = 0.1 + (0.3 - 0.1) ./ (1 + exp(-params_struct.(field)));     
                elseif any(strcmp(field,{'decision_thresh_baseline'}))
                    params.(field) = .5 + (1000 - .5) ./ (1 + exp(-params_struct.(field)));
                elseif any(strcmp(field,{'sigma_d', 'sigma_r'}))
                    params.(field) = (40) ./ (1 + exp(-params_struct.(field)));
                else 
                    disp(field);
                    error("Param not propertly transformed");
                end
            end
        end
    end
    varargout{1} = params;
    varargout{2} = prior_sigma;