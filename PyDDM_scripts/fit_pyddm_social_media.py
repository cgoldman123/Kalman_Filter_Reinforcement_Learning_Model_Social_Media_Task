import pyddm
from pyddm import Fitted 
import pandas as pd
import numpy as np
from KF_DDM_model import KF_DDM_model
from scipy.io import savemat

## Todo - be able to pass in fixed values like initial_sigma, starting value, reward sensitivity, max_rt, etc
# plot pyddm.plot.model_gui(model_to_fit, conditions={"deltaq": [-1, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, 1]})


########### Load in Social Media data and format as Sample object ###########
with open("L:/rsmith/lab-members/cgoldman/Wellbeing/social_media/output/test/60f8cc430a3d06de34a78773_beh_Like_04_01_25_T10-24-15.csv", "r") as f:
    df = pd.read_csv(f)
result_dir = f"L:/rsmith/lab-members/cgoldman/Wellbeing/social_media/output/test/60f8cc430a3d06de34a78773_like"  # directory to save results

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
social_media_df_clean = social_media_df.dropna(subset=["c"])
# Replace NA values in rt col with -1 so it can be passed into sample object
social_media_df_clean.loc[:, 'rt'] = social_media_df_clean['rt'].fillna(-1)


social_media_sample = pyddm.Sample.from_pandas_dataframe(social_media_df_clean, rt_column_name="rt", choice_column_name="c", choice_names=("right", "left")) # note the ordering here is intentional since pyddm codes the first choice as 1 (upper) and the second as 0 (lower) which matches our coding left as 0 and right as 1

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
        model.sigma1 = model_stats["sigma1"] # sigma1 is the uncertainty of the left bandit
        model.sigma2 = model_stats["sigma2"] # sigma2 is the uncertainty of the right bandit
        model.relative_uncertainty_of_choice = model_stats["relative_uncertainty_of_choice"]
        model.total_uncertainty = model_stats["total_uncertainty"]
        model.change_in_uncertainty_after_choice = model_stats["change_in_uncertainty_after_choice"]


        return -likelihood




## SIMULATE DATA WITHOUT FITTING
# model_to_sim = pyddm.gddm(drift=lambda drift_reward_diff_mod,reward_diff,sigma_d,sigma_r,baseline_noise,side_bias,directed_exp,baseline_info_bonus,random_exp : drift_reward_diff_mod * reward_diff,
#                           noise=1.0, bound="B", nondecision=0, starting_position="x0", T_dur=1000,
#                           conditions=["game_number", "gameLength", "trial", "r", "reward_diff","left_bandit_reward","right_bandit_reward"],
#                           parameters={"drift_reward_diff_mod": .4, "B": 10, "x0": 0, "sigma_d": 8, "sigma_r": 8, "baseline_noise": 1, "side_bias": 1, "directed_exp": 1, "baseline_info_bonus": -1, "random_exp": 2}, choice_names=("right","left"))

# model_stats = KF_DDM_model(social_media_sample,model_to_sim,fit_or_sim="sim")


## FIT MODEL TO ACTUAL DATA
# This is kind of hacky, but we pass in the learning parameters as arguments to drift even though they aren't used for it. This is just so the loss function has access to them
model_to_fit = pyddm.gddm(drift=lambda drift_reward_diff_mod,reward_diff,sigma_d,sigma_r,baseline_noise,side_bias,directed_exp,baseline_info_bonus,random_exp : drift_reward_diff_mod * reward_diff,
                          noise=1.0, bound="B", nondecision=0, starting_position="x0", T_dur=4.17,
                          conditions=["game_number", "gameLength", "trial", "r", "reward_diff"],
                          parameters={"drift_reward_diff_mod": (-2,2), "B": (0.3, 2), "x0": (-.8, .8), "sigma_d": (0,20), "sigma_r": (0,20), "baseline_noise": (1,10), "side_bias": (-2,2), "directed_exp": (-5,5), "baseline_info_bonus": (-5,5), "random_exp": (-5,5)}, choice_names=("right","left"))

model_to_fit.fit(sample=social_media_sample, lossfunction=KF_DDM_Loss)



# Save fit parameter estimates
# model_to_fit.get_fit_result()
params = model_to_fit.parameters()
# Extract the fitted parameters into a flat dictionary for easier access
fit_result = {
    f"{name}": float(val) if isinstance(val, Fitted) else val
    for comp, subdict in params.items()
    for name, val in subdict.items()
}
# Extract the model accuracy and average action probability
fit_result["average_action_prob"] = np.nanmean(model_to_fit.action_probs)
fit_result["average_action_prob_H1_1"] = np.nanmean(model_to_fit.action_probs[::2,4])
fit_result["average_action_prob_H5_1"] = np.nanmean(model_to_fit.action_probs[1::2,4])
fit_result["average_action_prob_H1_2"] = np.nanmean(model_to_fit.action_probs[:,5])
fit_result["average_action_prob_H1_3"] = np.nanmean(model_to_fit.action_probs[:,6])
fit_result["average_action_prob_H1_4"] = np.nanmean(model_to_fit.action_probs[:,7])
fit_result["average_action_prob_H1_5"] = np.nanmean(model_to_fit.action_probs[:,8])
fit_result["model_acc"] = np.nanmean(model_to_fit.model_acc)
fit_result["final_loss"] = pyddm.get_model_loss(model=model_to_fit, sample=social_media_sample, lossfunction=KF_DDM_Loss)
# Store the number of invalid trials
fit_result["num_invalid_rts"] = model_to_fit.num_invalid_rts

# Store the model output
model_output = {
    "action_probs": model_to_fit.action_probs,
    "model_acc": model_to_fit.model_acc,
    "exp_vals": model_to_fit.exp_vals,
    "pred_errors": model_to_fit.pred_errors,
    "pred_errors_alpha": model_to_fit.pred_errors_alpha,
    "alpha": model_to_fit.alpha,
    "sigma1": model_to_fit.sigma1,
    "sigma2": model_to_fit.sigma2,
    "relative_uncertainty_of_choice": model_to_fit.relative_uncertainty_of_choice,
    "total_uncertainty": model_to_fit.total_uncertainty,
    "change_in_uncertainty_after_choice": model_to_fit.change_in_uncertainty_after_choice,
}

model_output["actions"] = social_media_df["c"].values.reshape(40, 9) + 1
model_output["rewards"] = social_media_df["r"].values.reshape(40, 9)
model_output["rewards"] = social_media_df["r"].values.reshape(40, 9) # todo save RTs

# Save the fitted model before it gets written over during recoverability analysis
model_fit_to_data = model_to_fit
# Examine recoverability on fit parameters
# This is kind of hacky, but we pass in the learning parameters as arguments to drift even though they aren't used for it. This is just so the loss function has access to them
model_to_sim = pyddm.gddm(drift=lambda drift_reward_diff_mod,reward_diff,sigma_d,sigma_r,baseline_noise,side_bias,directed_exp,baseline_info_bonus,random_exp : drift_reward_diff_mod * reward_diff,
                          noise=1.0, bound="B", nondecision=0, starting_position="x0", T_dur=4.17,
                          conditions=["game_number", "gameLength", "trial", "r", "reward_diff"],
                          parameters={"drift_reward_diff_mod": fit_result["drift_reward_diff_mod"], "B": fit_result["B"], "x0": fit_result["x0"], "sigma_d": fit_result["sigma_d"], "sigma_r": fit_result["sigma_r"], "baseline_noise": fit_result["baseline_noise"], "side_bias": fit_result["side_bias"], "directed_exp": fit_result["directed_exp"], "baseline_info_bonus": fit_result["baseline_info_bonus"], "random_exp": fit_result["random_exp"]}, choice_names=("right","left"))

simulated_behavior = KF_DDM_model(social_media_sample,model_to_sim,fit_or_sim="sim")

simulated_sample = pyddm.Sample.from_pandas_dataframe(simulated_behavior["data"], rt_column_name="RT", choice_column_name="choice", choice_names=("right", "left")) # note the ordering here is intentional since pyddm codes the first choice as 1 (upper) and the second as 0 (lower) which matches our coding left as 0 and right as 1

# Fit the simulated data
model_to_fit.fit(sample=simulated_sample, lossfunction=KF_DDM_Loss)
model_fit_to_simulated_data = model_to_fit
# Extract the parameter estimates for the model fit to the simulated data
fit_result.update({
    f"simfit_{name}": float(val) if isinstance(val, Fitted) else val
    for comp, subdict in params.items()
    for name, val in subdict.items()
})
fit_result["simfit_average_action_prob"] = np.nanmean(model_to_fit.action_probs)
fit_result["simfit_model_acc"] = np.nanmean(model_to_fit.model_acc)
fit_result["simfit_final_loss"] = pyddm.get_model_loss(model=model_to_fit, sample=social_media_sample, lossfunction=KF_DDM_Loss)


simfit_datastruct = {
    "actions": simulated_behavior["data"].choice,
    "rewards": simulated_behavior["data"].r,
    "RTs": simulated_behavior["data"].RT,
}
model_output["simfit_datastruct"] = simfit_datastruct
# Rename two of the dictionary fields because too long
model_output["rel_uncertainty"] = model_output.pop("relative_uncertainty_of_choice")
model_output["chge_uncertainty_af_choice"] = model_output.pop("change_in_uncertainty_after_choice")

print("hi")

# Save results
savemat(f"{result_dir}_model_results.mat", {
    "fit_result": fit_result,
    "model_output": model_output
})
