# KF DDM model file
# Define a function that returns the likelihood of data under a model and can simulate data based on parameters
import pandas as pd
import numpy as np
from pyddm.logger import logger
from jensen_shannon_divergence_quad_method import jsd_normal
import matplotlib.pyplot as plt # USed for plot in the model function
from pyddm import BoundConstant, Fitted, BoundCollapsingLinear # USed for debugging purposes when fixing parameters in the loss function


# Add a filter to the logger to check for the Renormalization warning, where the probability of hitting the upper, lower, or undecided boundary is not 1 so it had to be renormalized
had_renorm = False
def renorm_filter(record):
    global had_renorm               # ← declare you’re writing the module flag
    if "Renormalizing probability density" in record.getMessage():
        had_renorm = True
    return True

logger.addFilter(renorm_filter)


def KF_DDM_model(sample,model,fit_or_sim, sim_using_max_pdf=False):
    global had_renorm 

    settings = model.settings

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
    drift_dcsn_noise_mod = model.get_dependence("drift").drift_dcsn_noise_mod
    rel_uncert_mod = model.get_dependence("drift").rel_uncert_mod
    sigma_scaler = model.get_dependence("drift").sigma_scaler

    # Initialize variables to hold output
    G = 40 # Number of games
    initial_sigma = 10000
    reward_sensitivity = 1
    max_rt = 7 # Maximum reaction time in seconds

    rt_pdf = np.full((G, 10), np.nan)
    action_probs = np.full((G, 9), np.nan)
    model_acc = np.full((G, 9), np.nan)
    predicted_RT = np.full((G, 9), np.nan)
    abs_error_RT = np.full((G, 9), np.nan)

    pred_errors = np.full((G, 10), np.nan)
    pred_errors_alpha = np.full((G, 9), np.nan)
    exp_vals = np.full((G, 10), np.nan)
    alpha = np.full((G, 10), np.nan)

    # sigma1 and sigma2: initialize first column with `initial_sigma`, rest zeros
    sigma1 = np.hstack((initial_sigma * np.ones((G, 1)), np.zeros((G, 9)))) # uncertainty of left bandit
    sigma2 = np.hstack((initial_sigma * np.ones((G, 1)), np.zeros((G, 9)))) # uncertainty of right bandit

    total_uncertainty = np.full((G, 10), np.nan)
    rdiff_chosen_opt = np.full((G, 10), np.nan)
    jsd_diff_chosen_opt = np.full((G, 10), np.nan)
    
    relative_uncertainty_of_choice = np.full((G, 10), np.nan)
    change_in_uncertainty_after_choice = np.full((G, 10), np.nan)

    num_invalid_rts = 0

    # --- catch errors during model execution ---
    try:
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
                # Calculate the reward difference by subtracting the mean of the right vs left option
                reward_diff = mu2[trial_num] - mu1[trial_num] # reward difference between the two bandits
                # Calculate the jensen shannon divergence between the two bandits for the forced choices. Note this calculation will be written over for the free choices to account for the effect of random exploration.
                jsd_val = jsd_normal(mu1[trial_num], sigma1[game_num,trial_num]*baseline_noise, mu2[trial_num], sigma2[game_num,trial_num]*baseline_noise)

                if trial_num >= 4:
                    if trial['gameLength'] == 5: # horizon is 1
                        T = 0
                        Y = 1
                    else: # horizon is 5
                        T = directed_exp
                        Y = random_exp
                        

                    z = .5 # hyperparam controlling steepness of curve

                    # Exponential descent from T to 0 over trials within a game
                    # info_bonus_bandit1 = sigma1[game_num,trial_num]*baseline_info_bonus + sigma1[game_num,trial_num]*T*(np.exp(-z*(trial_num-4))-np.exp(-4*z))/(1-np.exp(-4*z))
                    # info_bonus_bandit2 = sigma2[game_num,trial_num]*baseline_info_bonus + sigma2[game_num,trial_num]*T*(np.exp(-z*(trial_num-4))-np.exp(-4*z))/(1-np.exp(-4*z))

                    relative_uncertainty = (sigma2[game_num,trial_num] - sigma1[game_num,trial_num])
                    # If the number of times chosen the left side is higher than the right side, then higher values of baseline_info_bonus and directed exploration should increase the probability of choosing the right side
                    subset = data.loc[(data["game_number"] == game_numbers[game_num]) & (data["trial"] <= trial_num)]
                    num_choose_left = (subset['choice'] == 0).sum()
                    num_choose_right = (subset['choice'] == 1).sum()
                    if (num_choose_left - num_choose_right) > 0:
                        info_diff = relative_uncertainty*rel_uncert_mod  + baseline_info_bonus + T*(np.exp(-z*(trial_num-4))-np.exp(-4*z))/(1-np.exp(-4*z))
                    elif (num_choose_left - num_choose_right) < 0:
                        info_diff = relative_uncertainty*rel_uncert_mod  - baseline_info_bonus - T*(np.exp(-z*(trial_num-4))-np.exp(-4*z))/(1-np.exp(-4*z))
                    else:
                        info_diff = relative_uncertainty*rel_uncert_mod

                    
                    # Exponential descent from Y to 1 over trials within a game
                    RE = Y + ((1 - Y) * (1 - np.exp(-z * (trial_num - 4))) / (1 - np.exp(-4 * z)))
                    
                    # Calculate the jensen shannon divergence between the reward distributions of the two bandits
                    # Note that jsd_val is in nats, so it's pretty small. We multiply by 10 to make it have a bigger effect on the drift value.
                    jsd_val = jsd_normal(mu1[trial_num], sigma1[game_num,trial_num]*baseline_noise*RE, mu2[trial_num], sigma2[game_num,trial_num]*baseline_noise*RE)

                    # decision_noise = total_uncertainty[game_num,trial_num]*baseline_noise*RE

                    # Transform the starting position value so it's between -1 and 1. May want to smooth out function
                    starting_position_value =  np.tanh((info_diff+side_bias)/1)
                    # Get the drift_value by combining the reward difference and decision noise. The decision noise will push the drift in opposite direction of the reward difference. 
                    if reward_diff > 0:
                        # drift_value = (drift_rwrd_diff_mod * reward_diff) - (drift_dcsn_noise_mod * decision_noise)
                        if "Use_JSD" in settings:
                            # Divide by ln(2) since that's the max jensen shannon divergence value
                            drift_value = jsd_val/np.log(2)
                        elif "Use_reward_difference" in settings:
                            drift_value = (reward_diff/baseline_noise) - total_uncertainty[game_num,trial_num]*RE
                            
                    else:
                        # drift_value = (drift_rwrd_diff_mod * reward_diff) + (drift_dcsn_noise_mod * decision_noise)
                        if "Use_JSD" in settings:
                        # Divide by ln(2) since that's the max jensen shannon divergence value
                            drift_value = -jsd_val/np.log(2)
                        elif "Use_reward_difference" in settings:
                            drift_value = (reward_diff/baseline_noise) + total_uncertainty[game_num,trial_num]*RE


                    if fit_or_sim == "fit":
                        choice = "left" if trial['choice'] == 0 else "right"

                        # Only consider reaction times less than the max rt in the log likelihood
                        if trial['RT'] < max_rt:
                            # solve a ddm (i.e., get the probability density function) for current DDM parameters
                            # Higher values of reward_diff and side_bias indicate greater preference for right bandit (band it 1 vs 0)
                            had_renorm = False

                            # REMEMBER TO COMMENT THIS OUT!!!
                            # model._bounddep = BoundCollapsingLinear(B=1.5, t=.5) 
                            # model._bounddep = BoundConstant(B=Fitted(1.5, minval=1.5, maxval=6))
                            # model._bounddep.bound_intercept = Fitted(3.1660782, minval=0, maxval=10) # This is the directed exploration
                            # model._bounddep.bound_slope = Fitted(.5, minval=0, maxval=10) # This is the directed exploration

                            sol = model.solve_numerical_c(
                                conditions={
                                    "drift_value": drift_value,
                                    "starting_position_value": starting_position_value
                                }
                            )
                            # plt.plot(sol.t_domain, sol.pdf("right"))
                            # plt.axvline(x=trial['RT'], color='red', linestyle='--', label='RT')
                            # plt.legend()  # Optional, if you want the 'RT' label to show up
                            # plt.show()

                            # Check to see if the renormalization warning was triggered
                            if had_renorm:
                                print("The mode had to renormalize the probability density function. Please check the model parameters:")
                                params = model.parameters()
                                for subdict in params.values():
                                    for name, val in subdict.items():
                                        print(f"{name} = {float(val)}")
                                print("drift_value = ", drift_value)
                                print("starting_position_value = ", starting_position_value)
                                print()


                            # Evaluate the pdf of the reaction time for the chosen option. Note that left will be the bottom boundary and right upper
                            p = sol.evaluate(trial['RT'], choice)
                            assert p >= 0, "Probability density of a reaction time must be non-negative"
                            # Store the probability of the choice under the model
                            rt_pdf[game_num, trial_num] = p
                            # Store the action probability of the choice under the model
                            action_probs[game_num, trial_num] = sol.prob(choice)
                            model_acc[game_num, trial_num] = sol.prob(choice) > .5

                            # Get the absolute error between the model's predicted reaction time and the actual reaction time
                            if max(sol.pdf("left")) > max(sol.pdf("right")):
                                # Get the index of the max pdf
                                predicted_rt_index = int(np.argmax(sol.pdf("left")))
                            else:
                                predicted_rt_index = int(np.argmax(sol.pdf("right")))
                            predicted_RT[game_num, trial_num] = model.dt * predicted_rt_index
                            abs_error_RT[game_num, trial_num] = abs(trial['RT'] - predicted_RT[game_num, trial_num])

                        else:
                            num_invalid_rts += 1
                    else:
                        # Simulate a reaction time from the model based on the parameters

                        # solve a ddm (i.e., get the probability density function) for current DDM parameters
                        # Higher values of reward_diff and side_bias indicate greater preference for right bandit (band it 1 vs 0)
                        sol = model.solve_numerical_c(conditions={"drift_value": drift_value,
                                                                    "starting_position_value": starting_position_value})   
                        
                        # print(f"RE: ",RE)                    
                        # print(f"Drift: ",drift_value)                    
                        # print(f"Starting position: ",starting_position_value)                    
                        # print(f"Prob choose left: ",sol.prob("left"))
                        # Use the reaction time with the max probability density
                        if sim_using_max_pdf:
                            # If the left choice was more likely
                            if max(sol.pdf("left")) > max(sol.pdf("right")):
                                # Get the index of the max pdf
                                rt_index = int(np.argmax(sol.pdf("left")))
                                choice = 0
                            else:
                                rt_index = int(np.argmax(sol.pdf("right")))
                                choice = 1
                            # Get the reaction time at that index
                            RT = model.dt * rt_index
                            # Store the simulated action in a dataframe
                            res = pd.DataFrame({'choice': [choice], 'RT': [RT]})

                        # Sample from the pdf distributions
                        else:
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
                                    
                # Increase uncertainty before the first free choice
                if trial_num == 3:
                    sigma1[game_num, trial_num + 1] *= sigma_scaler
                    sigma2[game_num, trial_num + 1] *= sigma_scaler


                # Save the total uncertainty and reward difference for the chosen option
                # total uncertainty is variance of both arms
                total_uncertainty[game_num,trial_num] = (sigma1[game_num,trial_num]**2 + sigma2[game_num,trial_num]**2)**.5
                # Reward difference is mu2 (right) - mu1 (left) so when the person chooses the left bandit, the reward difference is negative
                rdiff_chosen_opt[game_num, trial_num] = reward_diff if trial['choice'] == 1 else -reward_diff
                # The jensen shannon divergence is always positive, so we multiply it by -1 (i.e., model the person choosing the worse side) if the person chose the left bandit when (mu2 > mu1) or the right bandit when (mu1 > mu2)
                jsd_diff_chosen_opt[game_num, trial_num] = jsd_val if (trial['choice'] == 1 and reward_diff > 0) or (trial['choice'] == 0 and reward_diff < 0) else -jsd_val


    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Printing the parameter values: ")
        for subdict in model.parameters().values():
            for name, val in subdict.items():
                print(f"  {name} = {val}")
        # Raise the exception so we still get traceback
        raise

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
        "predicted_RT": predicted_RT,
        "abs_error_RT": abs_error_RT,
        "rdiff_chosen_opt":rdiff_chosen_opt, # Reward difference of the chosen option
        "jsd_diff_chosen_opt":jsd_diff_chosen_opt, # JSD difference of the chosen option
        "data": data,
    }
