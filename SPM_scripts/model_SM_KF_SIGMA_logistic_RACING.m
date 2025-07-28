function model_output = model_SM_KF_SIGMA_logistic_RACING(params, actions_and_rts, rewards, mdp, sim)
    dbstop if error;
    % note that right bandit: 
    % mu2
    % c=2
    % free choice = 1
    % actions = 2
    G = mdp.G; % num of games
    
    max_rt = mdp.settings.max_rt;
    %% for testing %%
    %%%%%%
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
    wd = params.wd;
    ws = params.ws;
    V0 = params.V0;
    
   
    
    % initialize variables

    actions = actions_and_rts.actions;
    rts = actions_and_rts.RTs;
    rt_pdf = nan(G,9);
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

    num_invalid_rts = 0;

    decision_thresh = nan(G,9);

    
    for g = 1:G  % loop over games
        % values
        mu1 = [initial_mu nan nan nan nan nan nan nan nan];
        mu2 = [initial_mu nan nan nan nan nan nan nan nan];
    
        % learning rates 
        alpha1 = nan(1,9); 
        alpha2 = nan(1,9); 
    
        % pick H1 vs H5 params
        if mdp.C1(g) == 1
            info_bonus     = h1_info_bonus;
            decision_noise = h1_dec_noise;
            side_bias      = side_bias_h1;
        else
            info_bonus     = h5_info_bonus;
            decision_noise = h5_dec_noise;
            side_bias      = side_bias_h5;
        end
    
        for t = 1:5  % 4 forced + 1 free choice
            if t == 5
                % compute trial‐specific predictors
                reward_diff = mu2(t) - mu1(t);
                info_diff   = mdp.dI(g);
                p = 1 / (1 + exp((reward_diff + info_diff*info_bonus + side_bias) / decision_noise));
    
                %DDM PARAM MAPPING
                % drift
                drift = params.drift_baseline;
                if any(contains(mdp.settings.drift_mapping,'reward_diff'))
                    drift = drift + params.drift_reward_diff_mod * reward_diff;
                end
                if any(contains(mdp.settings.drift_mapping,'info_diff'))
                    drift = drift + info_diff * info_bonus;
                end
                if any(contains(mdp.settings.drift_mapping,'side_bias'))
                    drift = drift + side_bias;
                end
                if any(contains(mdp.settings.drift_mapping,'decision_noise'))
                    drift = drift / decision_noise;
                end
    
                % starting bias (0–1)
                sbias = log(params.starting_bias_baseline/(1 - params.starting_bias_baseline));
                if any(contains(mdp.settings.bias_mapping,'reward_diff'))
                    sbias = sbias + params.starting_bias_reward_diff_mod * reward_diff;
                end
                if any(contains(mdp.settings.bias_mapping,'info_diff'))
                    sbias = sbias + info_diff * info_bonus;
                end
                if any(contains(mdp.settings.bias_mapping,'side_bias'))
                    sbias = sbias + side_bias;
                end
                starting_bias = 1/(1 + exp(-sbias));
    
                % threshold
                decision_thresh(g,t) = params.decision_thresh_baseline;
                if any(contains(mdp.settings.thresh_mapping,'decision_noise'))
                    decision_thresh(g,t) = decision_noise;
                end
                
                %BUILD 2‑ACCUMULATOR RACE%
                % drifts: Based on Advantaged Racing Diffusion 
                v1 = V0 + wd*(mu1(t) - mu2(t)) + ws*(mu1(t) + mu2(t));
                v2 = V0 + wd*(mu2(t) - mu1(t)) +  ws*(mu1(t) + mu2(t));
                v_cell = {v1, v2};
                % across‐trial noise scaling variability. Just dummy values
                % for now
                s_cell = {1, 1};
      
                B1     = decision_thresh(g,t); % not sure if we're using starting bias * starting_bias;
                B2     = B1;
                B_cell = { B1, B2};
                % no trial‐to‐trial threshold variability for now
                A_cell =  {0, 0};
                t0_i   = {0,0}; % No non-decision time for now
    
                %SIMULATION OR FITTING
                if sim
                    % call the race model for 1 trial
                    out = rWaldRace(1, ...      % simulate 1 trial
                                    v_cell, ... % 1×2 cell of drifts
                                    B_cell, ... % 1×2 cell of thresholds
                                    A_cell, ... % 1×2 cell of threshold variabilities
                                    t0_i,   ... % scalar non‑decision time
                                    s_cell);    % 1×2 cell of noise scales
                
                    % extract simulated RT and choice
                    simmed_rt   = out.RT;
                    chose_right = out.R;        % 1 or 2
                
                    actions(g,t) = chose_right + 0;  % match your existing coding
                    if chose_right == 2
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
            if actions(g,t) == 1
                % left bandit update
                relative_uncertainty_of_choice(g,t)    = sigma1(g,t) - sigma2(g,t);
                tmp = 1/(sigma1(g,t)^2 + sigma_d^2) + 1/(sigma_r^2);
                sigma1(g,t+1) = sqrt(1/tmp);
                change_in_uncertainty_after_choice(g,t) = sigma1(g,t+1) - sigma1(g,t);
                alpha1(t)     = (sigma1(g,t+1)/sigma_r)^2;
    
                sigma2(g,t+1) = sqrt(sigma2(g,t)^2 + sigma_d^2);
    
                exp_vals(g,t)       = mu1(t);
                pred_errors(g,t)    = reward_sensitivity*rewards(g,t) - exp_vals(g,t);
                alpha(g,t)          = alpha1(t);
                pred_errors_alpha(g,t) = alpha1(t)*pred_errors(g,t);
                mu1(t+1)            = mu1(t) + pred_errors_alpha(g,t);
                mu2(t+1)            = mu2(t);
            else
                % right bandit update
                relative_uncertainty_of_choice(g,t)    = sigma2(g,t) - sigma1(g,t);
                tmp = 1/(sigma2(g,t)^2 + sigma_d^2) + 1/(sigma_r^2);
                sigma2(g,t+1) = sqrt(1/tmp);
                change_in_uncertainty_after_choice(g,t) = sigma2(g,t+1) - sigma2(g,t);
                alpha2(t)     = (sigma2(g,t+1)/sigma_r)^2;
    
                sigma1(g,t+1) = sqrt(sigma1(g,t)^2 + sigma_d^2);
    
                exp_vals(g,t)       = mu2(t);
                pred_errors(g,t)    = reward_sensitivity*rewards(g,t) - exp_vals(g,t);
                alpha(g,t)          = alpha2(t);
                pred_errors_alpha(g,t) = alpha2(t)*pred_errors(g,t);
                mu2(t+1)            = mu2(t) + pred_errors_alpha(g,t);
                mu1(t+1)            = mu1(t);
            end
    
            if ~sim
                if rts(g,t) <= 0 || rts(g,t) >= max_rt
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
    model_output.change_in_uncertainty_after_choice = change_in_uncertainty_after_choice;

end