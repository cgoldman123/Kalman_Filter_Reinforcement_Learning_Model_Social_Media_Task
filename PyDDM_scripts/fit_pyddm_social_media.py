import pyddm
from pyddm import Fitted 
import pandas as pd
import numpy as np


########### Load in Social Media data and format as Sample object ###########
with open("L:/rsmith/lab-members/cgoldman/Wellbeing/social_media/output/test/60f8cc430a3d06de34a78773_beh_Like_04_01_25_T10-24-15.csv", "r") as f:
    df = pd.read_csv(f)


# Extract trial numbers
trial_nums = list(range(1, 10)) # A list from 1 to 9

# Create a tidy DataFrame by stacking trial-wise columns
social_media_df = pd.DataFrame({
    'game_number': pd.Series(range(1, 41)).repeat(9).values, # game number (1 to 40)
    'gameLength': df['gameLength'].repeat(9).values, # game length (5 or 9)
    'trial': trial_nums * len(df), # trial number (1 indexed)
    'r': df[[f'r{i}' for i in trial_nums]].values.flatten(), # observed reward
    'c': df[[f'c{i}' for i in trial_nums]].values.flatten() - 1, # choice: 0 for left, 1 for right
    'rt': df[[f'rt{i}' for i in trial_nums]].values.flatten(), # reaction time
    'left_bandit_outcome': df[[f'mu1_reward{i}' for i in trial_nums]].values.flatten(), # Outcome if the participant were to choose the left side. Note all subjects observed the same schedule.
    'right_bandit_outcome': df[[f'mu2_reward{i}' for i in trial_nums]].values.flatten(), # Outcome if the participant were to choose the right side. Note all subjects observed the same schedule.
})
social_media_df = social_media_df.dropna(subset=["c"])
# Replace NA values in rt col with -1 so it can be passed into sample object
social_media_df['rt'] = social_media_df['rt'].fillna(-1)


social_media_sample = pyddm.Sample.from_pandas_dataframe(social_media_df, rt_column_name="rt", choice_column_name="c", choice_names=("right", "left")) # note the ordering here is intentional since pyddm codes the first choice as 1 and the second as 0 which matches our coding left as 0 and right as 1

# Example functions to get summary statistics. Not that these functions use all trials, including forced choices
social_media_sample.cdf("left",dt=.01,T_dur=2)
social_media_sample.condition_names()
social_media_sample.condition_values("game_number")
social_media_sample.items(choice="left")
social_media_sample.pdf("left", dt=.01, T_dur=2)
social_media_sample.prob("left")




class KF_DDM_Loss(pyddm.LossFunction):
    name = "KF_DDM_Loss"
    def loss(self, model):
        model_stats = KF_DDM_model(self.sample,model,fit_or_sim="fit")
        
        # get log likelihood
        rt_pdf = model_stats["rt_pdf"]
        likelihood = np.sum(np.log(rt_pdf[~np.isnan(rt_pdf)]))
        # Store model statistics
        model.action_probs = model_stats["action_probs"]
        model.rt_pdf = rt_pdf
        model.model_acc = model_stats["model_acc"]
        model.num_invalid_rts = model_stats["num_invalid_rts"]
        model.exp_vals = model_stats["exp_vals"]
        model.pred_errors = model_stats["pred_errors"]
        model.pred_errors_alpha = model_stats["pred_errors_alpha"]
        model.alpha = model_stats["alpha"]
        model.sigma1 = model_stats["sigma1"]
        model.sigma2 = model_stats["sigma2"]
        model.relative_uncertainty_of_choice = model_stats["relative_uncertainty_of_choice"]
        model.total_uncertainty = model_stats["total_uncertainty"]
        model.change_in_uncertainty_after_choice = model_stats["change_in_uncertainty_after_choice"]


        return -likelihood


# Define a function that returns the likelihood of data under a model and can simulate data based on parameters
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
    action_probs = np.full((G, 10), np.nan)
    model_acc = np.full((G, 10), np.nan)

    pred_errors = np.full((G, 10), np.nan)
    pred_errors_alpha = np.full((G, 10), np.nan)
    exp_vals = np.full((G, 10), np.nan)
    alpha = np.full((G, 10), np.nan)

    # sigma1 and sigma2: initialize first column with `initial_sigma`, rest zeros
    sigma1 = np.hstack((initial_sigma * np.ones((G, 1)), np.zeros((G, 9))))
    sigma2 = np.hstack((initial_sigma * np.ones((G, 1)), np.zeros((G, 9))))

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
                    
                reward_diff = mu1[trial_num] - mu2[trial_num]

                z = .5 # hyperparam controlling steepness of curve

                # Exponential descent
                info_bonus_bandit1 = sigma1[game_num,trial_num]*baseline_info_bonus + sigma1[game_num,trial_num]*T*(np.exp(-z*(trial_num-5))-np.exp(-4*z))/(1-np.exp(-4*z))
                info_bonus_bandit2 = sigma2[game_num,trial_num]*baseline_info_bonus + sigma2[game_num,trial_num]*T*(np.exp(-z*(trial_num-5))-np.exp(-4*z))/(1-np.exp(-4*z))

                info_diff = info_bonus_bandit1 - info_bonus_bandit2

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
                        sol = model.solve_analytical(conditions={"reward_diff": reward_diff + side_bias})
                        # Evaluate the pdf of the reaction time for the chosen option. Note that left will be the bottom boundary and right upper
                        p = sol.evaluate(trial['RT'], choice)
                        assert p >= 0, "Probability density of a reaction time must be non-negative"
                        # Store the probability of the choice under the model
                        rt_pdf[game_num, trial_num] = p
                        # Store the action probability of the choice under the model
                        action_probs[game_num, trial_num] = sol.prob(choice)
                        model_acc[game_num, trial_num] = sol.prob(choice) > 1
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
                    # It seems like sol.sample() returns a choice where 0 corresponds to right and 1 to left, so we flip this
                    res.loc[0, "choice"] = 1 - res.loc[0, "choice"]

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


model_to_sim = pyddm.gddm(drift=lambda drift_reward_diff_mod,reward_diff,sigma_d,sigma_r,baseline_noise,side_bias,directed_exp,baseline_info_bonus,random_exp : drift_reward_diff_mod * reward_diff,
                          noise=1.0, bound="B", nondecision=0, starting_position="x0", T_dur=1000,
                          conditions=["game_number", "gameLength", "trial", "r", "reward_diff","left_bandit_reward","right_bandit_reward"],
                          parameters={"drift_reward_diff_mod": .4, "B": 10, "x0": 0, "sigma_d": 8, "sigma_r": 8, "baseline_noise": 1, "side_bias": 1, "directed_exp": 1, "baseline_info_bonus": -1, "random_exp": 2}, choice_names=("right","left"))

model_stats = KF_DDM_model(social_media_sample,model_to_sim,fit_or_sim="sim")


# This is kind of hacky, but we pass in the learning parameters as arguments to drift even though they aren't used for it. This is just so the loss function has access to them
model_to_fit = pyddm.gddm(drift=lambda drift_reward_diff_mod,reward_diff,sigma_d,sigma_r,baseline_noise,side_bias,directed_exp,baseline_info_bonus,random_exp : drift_reward_diff_mod * reward_diff,
                          noise=1.0, bound="B", nondecision=0, starting_position="x0", T_dur=4.17,
                          conditions=["game_number", "gameLength", "trial", "r", "reward_diff"],
                          parameters={"drift_reward_diff_mod": (-2,2), "B": (0.3, 2), "x0": (-.8, .8), "sigma_d": (0,20), "sigma_r": (0,20), "baseline_noise": (1,10), "side_bias": (-2,2), "directed_exp": (-5,5), "baseline_info_bonus": (-5,5), "random_exp": (-5,5)}, choice_names=("right","left"))



model_to_fit.fit(sample=social_media_sample, lossfunction=KF_DDM_Loss)





# To get the model statistics for the last round of fitting? run this:
pyddm.get_model_loss(model=model_to_fit, sample=social_media_sample, lossfunction=KF_DDM_Loss)
model_to_fit.action_probs
model_to_fit.rt_pdf
model_to_fit.model_acc
model_to_fit.num_invalid_rts
model_to_fit.exp_vals
model_to_fit.pred_errors
model_to_fit.pred_errors_alpha
model_to_fit.alpha
model_to_fit.sigma1
model_to_fit.sigma2
model_to_fit.relative_uncertainty_of_choice
model_to_fit.total_uncertainty
model_to_fit.change_in_uncertainty_after_choice

# todo PLOT
# pyddm.plot.model_gui(model_to_fit, conditions={"deltaq": [-1, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, 1]})

model_to_fit.get_fit_result()
params = model_to_fit.parameters()
# Extract the fitted parameters into a flat dictionary for easier access
flat_params = {
    f"{comp}.{name}": float(val) if isinstance(val, Fitted) else val
    for comp, subdict in params.items()
    for name, val in subdict.items()
}

# Examine recoverability on fit parameters
# This is kind of hacky, but we pass in the learning parameters as arguments to drift even though they aren't used for it. This is just so the loss function has access to them
model_to_sim = pyddm.gddm(drift=lambda drift_reward_diff_mod,reward_diff,sigma_d,sigma_r,baseline_noise,side_bias,directed_exp,baseline_info_bonus,random_exp : drift_reward_diff_mod * reward_diff,
                          noise=1.0, bound="B", nondecision=0, starting_position="x0", T_dur=4.17,
                          conditions=["game_number", "gameLength", "trial", "r", "reward_diff"],
                          parameters={"drift_reward_diff_mod": flat_params["drift_reward_diff_mod"], "B": flat_params["B"], "x0": flat_params["x0"], "sigma_d": flat_params["sigma_d"], "sigma_r": flat_params["sigma_r"], "baseline_noise": flat_params["baseline_noise"], "side_bias": flat_params["side_bias"], "directed_exp": flat_params["directed_exp"], "baseline_info_bonus": flat_params["baseline_info_bonus"], "random_exp": flat_params["random_exp"]}, choice_names=("right","left"))



print("hi")

## Todo - be able to pass in fixed values like initial_sigma, starting value, reward sensitivity, max_rt, etc