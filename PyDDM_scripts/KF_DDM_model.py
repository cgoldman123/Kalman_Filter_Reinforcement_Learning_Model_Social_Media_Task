# KF DDM model file
# Define a function that returns the likelihood of data under a model and can simulate data based on parameters
import pandas as pd
import numpy as np

def KF_DDM_model(sample,model,fit_or_sim):
    data = sample.to_pandas_dataframe()
    data = data.sort_values(by=["game_number", "trial"]) # Sort the dataframe by game number and trial number

    game_numbers = data['game_number'].unique()

    sigma_d = model.get_dependence("drift").sigma_d
    sigma_r = model.get_dependence("drift").sigma_r
    baseline_noise = model.get_dependence("drift").baseline_noise
    side_bias = model.get_dependence("drift").side_bias
    directed_exp = model.get_dependence("drift").directed_exp
    baseline_info_bonus = model.get_dependence("drift").baseline_info_bonus
    random_exp = model.get_dependence("drift").random_exp

    # Initialize variables to hold output
    G = 40 # Number of games
    initial_sigma = 10000
    reward_sensitivity = 1
    max_rt = 4.17 # Maximum reaction time in seconds

    rt_pdf = np.full((G, 10), np.nan)
    action_probs = np.full((G, 9), np.nan)
    model_acc = np.full((G, 9), np.nan)

    pred_errors = np.full((G, 10), np.nan)
    pred_errors_alpha = np.full((G, 9), np.nan)
    exp_vals = np.full((G, 10), np.nan)
    alpha = np.full((G, 10), np.nan)

    # sigma1 and sigma2: initialize first column with `initial_sigma`, rest zeros
    sigma1 = np.hstack((initial_sigma * np.ones((G, 1)), np.zeros((G, 9)))) # uncertainty of left bandit
    sigma2 = np.hstack((initial_sigma * np.ones((G, 1)), np.zeros((G, 9)))) # uncertainty of right bandit

    total_uncertainty = np.full((G, 10), np.nan)
    relative_uncertainty_of_choice = np.full((G, 10), np.nan)
    change_in_uncertainty_after_choice = np.full((G, 10), np.nan)

    num_invalid_rts = 0

    decision_thresh = np.full((G, 10), np.nan)


    for game_num in range(0, len(game_numbers)):
        game_df = data[data["game_number"] == game_numbers[game_num]] # Subset the dataframe by game number
        # Initialize empty vectors for each game
        mu1 = np.full(10, np.nan)
        mu1[0] = 50 # Irrelevant when initial_sigma is 10000 since it makes the learning rate on the first trial 1
        mu2 = np.full(10, np.nan)
        mu2[0] = 50 # Irrelevant when initial_sigma is 10000 since it makes the learning rate on the first trial 1
        
        alpha1 = np.full(10, np.nan)
        alpha2 = np.full(10, np.nan)
    
        for trial_num in range(0,len(game_df)):
            trial = game_df.iloc[trial_num]
            # Loop over forced choice trials
            if trial_num >= 4:
                if trial['gameLength'] == 5: # horizon is 1
                    T = 0
                    Y = 1
                else: # horizon is 5
                    T = directed_exp
                    Y = random_exp
                    
                reward_diff = mu2[trial_num] - mu1[trial_num] # reward difference between the two bandits

                z = .5 # hyperparam controlling steepness of curve

                # Exponential descent
                info_bonus_bandit1 = sigma1[game_num,trial_num]*baseline_info_bonus + sigma1[game_num,trial_num]*T*(np.exp(-z*(trial_num-5))-np.exp(-4*z))/(1-np.exp(-4*z))
                info_bonus_bandit2 = sigma2[game_num,trial_num]*baseline_info_bonus + sigma2[game_num,trial_num]*T*(np.exp(-z*(trial_num-5))-np.exp(-4*z))/(1-np.exp(-4*z))

                info_diff = info_bonus_bandit2 - info_bonus_bandit1 # information difference between the two bandits

                # total uncertainty is variance of both arms
                total_uncertainty[game_num,trial_num] = (sigma1[game_num,trial_num]**2 + sigma2[game_num,trial_num]**2)**.5
                
                # Exponential descent
                RE = Y + ((1 - Y) * (1 - np.exp(-z * (trial_num - 5))) / (1 - np.exp(-4 * z)))
                
                decision_noise = total_uncertainty[game_num,trial_num]*baseline_noise*RE


                if fit_or_sim == "fit":
                    choice = "left" if trial['choice'] == 0 else "right"
                    # Only consider reaction times less than the max rt in the log likelihood
                    if trial['RT'] < max_rt:
                        # solve a ddm (i.e., get the probability density function) for current DDM parameters
                        # Higher values of reward_diff and side_bias indicate greater preference for right bandit (band it 1 vs 0)
                        sol = model.solve_analytical(conditions={"reward_diff": reward_diff + side_bias})
                        # Evaluate the pdf of the reaction time for the chosen option. Note that left will be the bottom boundary and right upper
                        p = sol.evaluate(trial['RT'], choice)
                        assert p >= 0, "Probability density of a reaction time must be non-negative"
                        # Store the probability of the choice under the model
                        rt_pdf[game_num, trial_num] = p
                        # Store the action probability of the choice under the model
                        action_probs[game_num, trial_num] = sol.prob(choice)
                        model_acc[game_num, trial_num] = sol.prob(choice) > .5
                    else:
                        num_invalid_rts += 1
                else:
                    # Simulate a reaction time from the model based on the parameters
                    sol = model.solve_analytical(conditions={"reward_diff": reward_diff + side_bias})
                    res = sol.sample(1).to_pandas_dataframe(drop_undecided=True) # We only use the first non-undecided trial
                    # dataframe will be empty if the trial was undecided. Simulate again
                    max_num_simulations = 10
                    while res.empty:
                        if max_num_simulations == 0:
                            raise ValueError("The model simulated an undecided trial after 10 attempts. Please check the model parameters.")
                        max_num_simulations -= 1
                        res = sol.sample(1).to_pandas_dataframe(drop_undecided=True) # We only use the first non-undecided trial

                    # Assign the simulated action and RT to the dataframe
                    data.loc[(data["game_number"] == game_numbers[game_num]) & (data["trial"] == trial.trial),"choice"] = res.choice[0]
                    data.loc[(data["game_number"] == game_numbers[game_num]) & (data["trial"] == trial.trial), "RT"] = res.RT[0]
                    # Assign the simulated reward to the dataframe based on the option chosen
                    # If the choice is 0, the left bandit outcome is used, otherwise the right bandit outcome is used
                    if res.choice[0] == 0:
                        data.loc[(data["game_number"] == game_numbers[game_num]) & (data["trial"] == trial.trial), "r"] = trial['left_bandit_outcome']
                    else:
                        data.loc[(data["game_number"] == game_numbers[game_num]) & (data["trial"] == trial.trial), "r"] = trial['right_bandit_outcome']
                    # float(res['RT'][r])
                    simulated_trial = data[(data["game_number"] == game_numbers[game_num]) & (data["trial"] == trial.trial)] 
                    trial = simulated_trial.squeeze() # Convert the dataframe to a series
            # left option chosen so mu1 updates
            if trial['choice'] == 0:
                # save relative uncertainty of choice
                relative_uncertainty_of_choice[game_num, trial_num] = sigma1[game_num, trial_num] - sigma2[game_num, trial_num]

                # update sigma and learning rate
                temp = 1 / (sigma1[game_num, trial_num]**2 + sigma_d**2) + 1 / (sigma_r**2)
                sigma1[game_num, trial_num + 1] = np.sqrt(1 / temp)
                change_in_uncertainty_after_choice[game_num, trial_num] = sigma1[game_num, trial_num + 1] - sigma1[game_num, trial_num]
                alpha1[trial_num] = (sigma1[game_num, trial_num + 1] / sigma_r)**2

                temp = sigma2[game_num, trial_num]**2 + sigma_d**2
                sigma2[game_num, trial_num + 1] = np.sqrt(temp)

                exp_vals[game_num, trial_num] = mu1[trial_num]
                pred_errors[game_num, trial_num] = (reward_sensitivity * trial['r']) - exp_vals[game_num, trial_num]
                alpha[game_num, trial_num] = alpha1[trial_num]
                pred_errors_alpha[game_num, trial_num] = alpha1[trial_num] * pred_errors[game_num, trial_num]
                mu1[trial_num + 1] = mu1[trial_num] + pred_errors_alpha[game_num, trial_num]
                mu2[trial_num + 1] = mu2[trial_num]

            else:
                # right bandit choice, so mu2 updates
                relative_uncertainty_of_choice[game_num, trial_num] = sigma2[game_num, trial_num] - sigma1[game_num, trial_num]

                temp = 1 / (sigma2[game_num, trial_num]**2 + sigma_d**2) + 1 / (sigma_r**2)
                sigma2[game_num, trial_num + 1] = np.sqrt(1 / temp)
                change_in_uncertainty_after_choice[game_num, trial_num] = sigma2[game_num, trial_num + 1] - sigma2[game_num, trial_num]
                alpha2[trial_num] = (sigma2[game_num, trial_num + 1] / sigma_r)**2

                temp = sigma1[game_num, trial_num]**2 + sigma_d**2
                sigma1[game_num, trial_num + 1] = np.sqrt(temp)

                exp_vals[game_num, trial_num] = mu2[trial_num]
                pred_errors[game_num, trial_num] = (reward_sensitivity * trial['r']) - exp_vals[game_num, trial_num]
                alpha[game_num, trial_num] = alpha2[trial_num]
                pred_errors_alpha[game_num, trial_num] = alpha2[trial_num] * pred_errors[game_num, trial_num]
                mu2[trial_num + 1] = mu2[trial_num] + pred_errors_alpha[game_num, trial_num]
                mu1[trial_num + 1] = mu1[trial_num]

    # Return a dictionary of model statistics
    return {
        "action_probs": action_probs,
        "rt_pdf": rt_pdf,
        "model_acc": model_acc,
        "num_invalid_rts": num_invalid_rts,
        "exp_vals": exp_vals,
        "pred_errors": pred_errors,
        "pred_errors_alpha": pred_errors_alpha,
        "alpha": alpha,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "relative_uncertainty_of_choice": relative_uncertainty_of_choice,
        "total_uncertainty": total_uncertainty,
        "change_in_uncertainty_after_choice": change_in_uncertainty_after_choice,
        "num_invalid_rts": num_invalid_rts,
        "data": data,
    }
