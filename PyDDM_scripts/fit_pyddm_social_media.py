import pyddm
from pyddm import Fitted 
import pandas as pd
import numpy as np
from KF_DDM_model import KF_DDM_model
from scipy.io import savemat
import sys, random
import pyddm.plot
import matplotlib.pyplot as plt
import pickle # only needed for debugging purposes when a model_stats object is loaded into the loss function
from pyddm import BoundConstant, Fitted, BoundCollapsingLinear # USed for debugging purposes when fixing parameters in the loss function

# Get arguments from the command line if they are passed in
if len(sys.argv) > 1:
    outpath_beh = sys.argv[1] # Behavioral file location
    results_dir = sys.argv[2] # Directory to save results
    id = sys.argv[3] # Subject ID
    room_type = sys.argv[4] # Room type (e.g., Like, Dislike)
    timestamp = sys.argv[5] # Timestamp (e.g., 04_16_25_T10-39-55)
    settings = sys.argv[6] # Settings for the model (e.g., "Use_JSD_fit_all_RTs", "Use_JSD_fit_3_RTs", "Use_z_score_fit_all_RTs", "Use_z_score_fit_3_RTs")

else:
    # Note you'll have to change both outpath_beh and id to fit another subject
    outpath_beh = f"L:/rsmith/lab-members/cgoldman/Wellbeing/social_media/output/SM_fits_PYDDM_test_PYDDM_06-03-2025_21-30-31/Like/model1/5590a34cfdf99b729d4f69dc_beh_Like_06_03_25_T21-31-11.csv" # Behavioral file location
    id = "5590a34cfdf99b729d4f69dc" # Subject ID
    results_dir = f"L:/rsmith/lab-members/cgoldman/Wellbeing/social_media/output/test/" # Directory to save results
    room_type = "Like" # Room type (e.g., Like, Dislike)
    timestamp = "current_timestamp" # Timestamp (e.g., 04_16_25_T10-39-55)
    settings = "fit_all_RTs" # Settings for the model (e.g., "Use_JSD_fit_all_RTs", "Use_JSD_fit_3_RTs", "Use_z_score_fit_all_RTs", "Use_z_score_fit_3_RTs"

log_path = f"{results_dir}{id}_{room_type}_{timestamp}_model_log.txt"
sys.stdout = open(log_path, "w", buffering=1)  # line-buffered so that it updates the file in real-time
sys.stderr = sys.stdout  # Also capture errors
print("Welcome to the pyddm model fitting script!")

# Set the random seed for reproducibility
seed = 23
np.random.seed(seed)
random.seed(seed)
print(f"Random seed set to {seed}")
eps = np.finfo(float).eps

########### Load in Social Media data and format as Sample object ###########
with open(outpath_beh, "r") as f:
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
social_media_df_clean = social_media_df.dropna(subset=["c"])
# Replace NA values in rt col with -1 so it can be passed into sample object
social_media_df_clean.loc[:, 'rt'] = social_media_df_clean['rt'].fillna(-1)


social_media_sample = pyddm.Sample.from_pandas_dataframe(social_media_df_clean, rt_column_name="rt", choice_column_name="c", choice_names=("right", "left")) # note the ordering here is intentional since pyddm codes the first choice as 1 (upper) and the second as 0 (lower) which matches our coding left as 0 and right as 1

# Example functions to get summary statistics. Not that these functions use all trials, including forced choices
# social_media_sample.cdf("left",dt=.01,T_dur=2)
# social_media_sample.condition_names()
# social_media_sample.condition_values("game_number")
# social_media_sample.items(choice="left")
# social_media_sample.pdf("left", dt=.01, T_dur=2)
# social_media_sample.prob("left")




class KF_DDM_Loss(pyddm.LossFunction):
    name = "KF_DDM_Loss"
    def loss(self, model):
        # Hardcode parameters for debugging purposes
        # model._driftdep.base_info_bonus = Fitted(0.3496305974, minval=-4, maxval=4)
        # model._driftdep.random_exp = Fitted(6.4630579, minval=0.1, maxval=10)
        # model._driftdep.rel_uncert_mod = Fitted(-0.06463354, minval=-1, maxval=1) # This is the relative uncertainty of the choice
        # model._driftdep.sigma_d = Fitted(9.676269288, minval=0, maxval=10) # This is the uncertainty of the left bandit
        # model._driftdep.sigma_r = Fitted(4.000000, minval=0, maxval=10) # This is the uncertainty of the right bandit
        # model._driftdep.side_bias = Fitted(0.034263500, minval=-4, maxval=4) # This is the side bias
        # model._driftdep.directed_exp = Fitted(-0.59829014, minval=-4, maxval=4) # This is the directed exploration
        # model._driftdep.base_noise = Fitted(3.1660782, minval=-4, maxval=4) # This is the directed exploration
        # model._bounddep = BoundConstant(B=Fitted(1.501095, minval=1.5, maxval=6))
        model_stats = KF_DDM_model(self.sample,model, fit_or_sim="fit")
        
        # Load in model_stats object if you want to use it for debugging purposes; Comment this out unless debugging!!
        # with open("model_stats.pkl", "rb") as f:
        #     model_stats = pickle.load(f)

        # get log likelihood
        rt_pdf = model_stats["rt_pdf"]
        action_probs = model_stats["action_probs"]
        abs_error_RT = model_stats["abs_error_RT"]
        eps = np.finfo(float).eps
        if "fit_all_RTs" in settings:
            likelihood = np.sum(np.log(rt_pdf[~np.isnan(rt_pdf)] + eps)) # add small value to avoid log(0)
            avg_abs_error_RT = np.mean((abs_error_RT[~np.isnan(abs_error_RT)]))
        elif "fit_3_RTs" in settings:
            first_3_rts = rt_pdf[:, 0:7]
            first_3_rts_log_pdf = np.sum(np.log(first_3_rts[~np.isnan(first_3_rts)])) # Fit only the first 3 RTs but the other choice probabilities
            choices_4_and_5 = action_probs[:, 7:9] # Fit the choice probabilities for the 4th and 5th choices
            choices_4_and_5_log_prob = np.sum(np.log(choices_4_and_5[~np.isnan(choices_4_and_5)]))
            likelihood = first_3_rts_log_pdf + choices_4_and_5_log_prob # Combine the two log likelihoods
            first_3_rts_abs_error =abs_error_RT[:, 0:7]
            avg_abs_error_RT = np.mean((first_3_rts_abs_error[~np.isnan(first_3_rts_abs_error)]))

        # Store model statistics
        model.action_probs = action_probs
        model.rt_pdf = rt_pdf
        model.abs_error_RT = abs_error_RT
        model.avg_abs_error_RT = avg_abs_error_RT
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
# model_to_sim = pyddm.gddm(drift=lambda drift_rwrd_diff_mod,drift_dcsn_noise_mod,drift_value,sigma_d,sigma_r,base_noise,side_bias,directed_exp,base_info_bonus,random_exp : drift_value,
#                           starting_position=lambda starting_position_value: starting_position_value, 
#                           noise=1.0, bound="B", nondecision=0, T_dur=1000,
#                           conditions=["game_number", "gameLength", "trial", "r", "drift_value","starting_position_value"],
#                           parameters={"drift_rwrd_diff_mod": .5, "drift_dcsn_noise_mod": .1,"B": 1, "sigma_d": 8, "sigma_r": 8, "base_noise": 2, "side_bias": 1, "directed_exp": 2, "base_info_bonus": 2, "random_exp": 2}, choice_names=("right","left"))

# simulated_model = KF_DDM_model(social_media_sample,model_to_sim,fit_or_sim="sim")


## FIT MODEL TO ACTUAL DATA
# This is kind of hacky, but we pass in the learning parameters as arguments to drift even though they aren't used for it. This is just so the loss function has access to them
print("Setting up the model to fit behavioral data")
model_to_fit = pyddm.gddm(drift=lambda rdiff_bias_mod, random_exp, bound_intercept, bound_shift, base_noise, cong_DE, incong_DE, cong_base_info_bonus, incong_base_info_bonus, sigma_d, sigma_r, side_bias, drift_value : drift_value,
                          starting_position=lambda starting_position_value: starting_position_value, 
                          noise=1.0,    bound=lambda bound_value: max(bound_value, eps),
                          nondecision="nondectime", T_dur=7,
                          conditions=["game_number", "gameLength", "trial", "r", "drift_value","starting_position_value", "bound_value"],
                          parameters={"rdiff_bias_mod": (-10, 10), "nondectime": (0,.3), "random_exp": (0,10), "bound_intercept": (.5,5), "bound_shift": (-30,30), "sigma_d": (0,10), "sigma_r": (4,16), "base_noise": (0.1,10), "side_bias": (-.99,.99), "cong_DE": (-10,10), "incong_DE": (-10,10), "cong_base_info_bonus": (-10,10), "incong_base_info_bonus": (-10,10)}, choice_names=("right","left"))
model_to_fit.settings = settings

# Note that to plot using this function, I would have to pass in the conditions that get assigned in the model function (i.e., starting_position_value and drift_value). We would only be able to view the pdf and reaction time for one combination of conditions, which is not ideal.
# pyddm.plot.plot_fit_diagnostics(model=model_to_fit, sample=social_media_sample)
# plt.savefig("roitman-fit.png")
# plt.show()

print("Fitting behavioral data")
model_to_fit.fit(sample=social_media_sample, lossfunction=KF_DDM_Loss,verbose=True)

# Save fit parameter estimates
# model_to_fit.get_fit_result()
params = model_to_fit.parameters()
# Extract the fitted parameters into a flat dictionary for easier access
fit_result = {}
for subdict in params.values():
    for name, val in subdict.items():
        if isinstance(val, Fitted):
            fit_result[f"post_{name}"] = float(val)
            fit_result[f"min_{name}"] = val.minval
            fit_result[f"max_{name}"] = val.maxval
        else:
            fit_result[f"fixed_{name}"] = val

# Extract the model accuracy and average action probability
fit_result["average_action_prob"] = np.nanmean(model_to_fit.action_probs)
fit_result["average_action_prob_H1_1"] = np.nanmean(model_to_fit.action_probs[::2,4])
fit_result["average_action_prob_H5_1"] = np.nanmean(model_to_fit.action_probs[1::2,4])
fit_result["average_action_prob_H5_2"] = np.nanmean(model_to_fit.action_probs[:,5])
fit_result["average_action_prob_H5_3"] = np.nanmean(model_to_fit.action_probs[:,6])
fit_result["average_action_prob_H5_4"] = np.nanmean(model_to_fit.action_probs[:,7])
fit_result["average_action_prob_H5_5"] = np.nanmean(model_to_fit.action_probs[:,8])
fit_result["model_acc"] = np.nanmean(model_to_fit.model_acc)
fit_result["final_loss"] = pyddm.get_model_loss(model=model_to_fit, sample=social_media_sample, lossfunction=KF_DDM_Loss)
# Store the number of invalid trials
fit_result["num_invalid_rts"] = model_to_fit.num_invalid_rts
fit_result["avg_abs_error_RT"] = model_to_fit.avg_abs_error_RT

# Store the model output
model_output = {
    "action_probs": model_to_fit.action_probs,
    "rt_pdf": model_to_fit.rt_pdf,
    "abs_error_RT": model_to_fit.abs_error_RT,
    "predicted_RT": model_to_fit.predicted_RT,
    "rdiff_chosen_opt":model_to_fit.rdiff_chosen_opt,
    "rt_pdf_entropy": model_to_fit.rt_pdf_entropy,
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
model_output["rts"] = social_media_df["rt"].values.reshape(40, 9) 

# Save the fitted model before it gets written over during recoverability analysis
model_fit_to_data = model_to_fit
# Examine recoverability on fit parameters
# This is kind of hacky, but we pass in the learning parameters as arguments to drift even though they aren't used for it. This is just so the loss function has access to them
print("Setting up the model to simulate behavioral data")
model_to_sim = pyddm.gddm(drift=lambda rdiff_bias_mod, random_exp, bound_intercept, bound_shift, base_noise, cong_DE, incong_DE, cong_base_info_bonus, incong_base_info_bonus, sigma_d, sigma_r, side_bias, drift_value : drift_value,
                          starting_position=lambda starting_position_value: starting_position_value, 
                          noise=1.0, bound=lambda bound_value: max(bound_value, eps),
                          nondecision="nondectime", T_dur=100,
                          conditions=["game_number", "gameLength", "trial", "r", "drift_value","starting_position_value","bound_value"],
                          parameters={
    "nondectime": fit_result["post_nondectime"],
    "rdiff_bias_mod": fit_result["post_rdiff_bias_mod"],
    "random_exp": fit_result["post_random_exp"],
    "bound_intercept": fit_result["post_bound_intercept"],
    "bound_shift": fit_result["post_bound_shift"],
    "sigma_d": fit_result["post_sigma_d"],
    "sigma_r": fit_result["post_sigma_r"],
    "base_noise": fit_result["post_base_noise"],
    "side_bias": fit_result["post_side_bias"],
    "cong_DE": fit_result["post_cong_DE"],
    "incong_DE": fit_result["post_incong_DE"],
    "cong_base_info_bonus": fit_result["post_cong_base_info_bonus"],
    "incong_base_info_bonus": fit_result["post_incong_base_info_bonus"]
}, choice_names=("right","left"))
model_to_sim.settings = settings


print("Simulating behavioral data")
simulated_behavior = KF_DDM_model(social_media_sample,model_to_sim,fit_or_sim="sim")

simulated_sample = pyddm.Sample.from_pandas_dataframe(simulated_behavior["data"], rt_column_name="RT", choice_column_name="choice", choice_names=("right", "left")) # note the ordering here is intentional since pyddm codes the first choice as 1 (upper) and the second as 0 (lower) which matches our coding left as 0 and right as 1

# Fit the simulated data
print("Fitting simulated behavioral data")
model_to_fit.fit(sample=simulated_sample, lossfunction=KF_DDM_Loss)
params = model_to_fit.parameters()
model_fit_to_simulated_data = model_to_fit
# Extract the parameter estimates for the model fit to the simulated data (hence prefix sft)
simfit_result = {}
for subdict in params.values():
    for name, val in subdict.items():
        if isinstance(val, Fitted):
            simfit_result[f"sft_post_{name}"] = float(val)
            simfit_result[f"sft_min_{name}"] = val.minval
            simfit_result[f"sft_max_{name}"] = val.maxval
        else:
            simfit_result[f"sft_fixed_{name}"] = val
fit_result.update(simfit_result)

fit_result["sft_average_action_prob"] = np.nanmean(model_to_fit.action_probs)
fit_result["sft_model_acc"] = np.nanmean(model_to_fit.model_acc)
fit_result["sft_final_loss"] = pyddm.get_model_loss(model=model_to_fit, sample=social_media_sample, lossfunction=KF_DDM_Loss)

# Reformat the simulated data to match the original data structure
simulated_data_raw = simulated_behavior["data"]
simulated_data_full = []
i = 0
while i < len(simulated_data_raw):
    game = simulated_data_raw.iloc[i]
    game_len = int(game['gameLength'])
    game_rows = simulated_data_raw.iloc[i:i+game_len]
    simulated_data_full.append(game_rows)

    if game_len == 5:
        game_number = game['game_number']
        padding = pd.DataFrame({
            'game_number': [game_number]*4,
            'gameLength': [5]*4,
            'trial': [6,7,8,9],
            'r': [np.nan]*4,
            'choice': [np.nan]*4,
            'RT': [np.nan]*4,
            'left_bandit_outcome': [np.nan]*4,
            'right_bandit_outcome': [np.nan]*4
        })
        simulated_data_full.append(padding)

    i += game_len

simulated_data_df = pd.concat(simulated_data_full).reset_index(drop=True)

# Build the datastruct for the simulated data
simfit_datastruct = {
    "actions": simulated_data_df.choice.values.reshape(40, 9) + 1,
    "rewards": simulated_data_df.r.values.reshape(40, 9),
    "RTs": simulated_data_df.RT.values.reshape(40, 9),
}
# Replace -1 valus from the RTs with NaNs
simfit_datastruct["RTs"][simfit_datastruct["RTs"] == -1] = np.nan


model_output["sft_datastruct"] = simfit_datastruct
# Rename two of the dictionary fields because too long
model_output["rel_uncertainty"] = model_output.pop("relative_uncertainty_of_choice")
model_output["chge_uncertainty_af_choice"] = model_output.pop("change_in_uncertainty_after_choice")


# Save results
savemat(f"{results_dir}{id}_{room_type}_{timestamp}_model_results_pyddm.mat", {
    "fit_result": fit_result,
    "model_output": model_output
})
print("Finished saving results. Exiting script!")
