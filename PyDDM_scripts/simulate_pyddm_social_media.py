import pyddm
from pyddm import Fitted 
import pandas as pd
import numpy as np
from KF_DDM_model import KF_DDM_model
from scipy.io import savemat
import sys, random
import pyddm.plot
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import sem

# Use this function to simulate model-free metrics under a range of parameter values

# Get arguments from the command line if they are passed in
if len(sys.argv) > 1:
    outpath_beh = sys.argv[1] # Behavioral file location
    results_dir = sys.argv[2] # Directory to save results
    id = sys.argv[3] # Subject ID
    room_type = sys.argv[4] # Room type (e.g., Like, Dislike)
    timestamp = sys.argv[5] # Timestamp (e.g., 04_16_25_T10-39-55)
else:
    # Note you'll have to change both outpath_beh and id to fit another subject
    outpath_beh = f"L:/rsmith/lab-members/cgoldman/Wellbeing/social_media/output/SM_fits_PYDDM_test_PYDDM_04-21-2025_15-04-20/Like/model1/568d0641b5a2c2000cb657d0_beh_Like_04_21_25_T15-57-57.csv" # Behavioral file location
    id = "568d0641b5a2c2000cb657d0" # Subject ID
    results_dir = f"L:/rsmith/lab-members/cgoldman/Wellbeing/social_media/output/test/" # Directory to save results
    room_type = "Like" # Room type (e.g., Like, Dislike)
    timestamp = "current_timestamp" # Timestamp (e.g., 04_16_25_T10-39-55)


log_path = f"{results_dir}{id}_{room_type}_{timestamp}_model_log.txt"
sys.stdout = open(log_path, "w", buffering=1)  # line-buffered so that it updates the file in real-time
sys.stderr = sys.stdout  # Also capture errors
print("Welcome to the pyddm model simulating script!")

# Set the random seed for reproducibility
seed = 23
np.random.seed(seed)
random.seed(seed)
print(f"Random seed set to {seed}")

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

# Create a sample of values to pass into the model
social_media_sample = pyddm.Sample.from_pandas_dataframe(social_media_df_clean, rt_column_name="rt", choice_column_name="c", choice_names=("right", "left")) # note the ordering here is intentional since pyddm codes the first choice as 1 (upper) and the second as 0 (lower) which matches our coding left as 0 and right as 1



# ------------------------------------------------------------------
# Generic function to compute the metric: “prob choose higher observed mean” for the following conditions:
#   • game_len   – 5 for H1 games, 9 for H5 games, etc.
#   • trial_idx  – which free-choice trial to score (5, 6, 7, …)
# ------------------------------------------------------------------
def high_obs_mean(sim_df: pd.DataFrame,
                  game_len: int,
                  trial_idx: int) -> dict:
    """Return probability of choosing the higher-mean bandit *and*
    the average RTs for trials where the ‘better’ vs ‘worse’ bandit
    was chosen."""

    # subset to one game length
    g = sim_df[sim_df["gameLength"] == game_len]

    # forced trials (1-4) → compute which side has the higher mean
    forced = g[g["trial"].isin([1, 2, 3, 4])]
    means = (forced
             .groupby("game_number")
             .agg(left_mean=('left_bandit_outcome',  'mean'),
                  right_mean=('right_bandit_outcome', 'mean')))
    means["better"] = np.where(means.left_mean  > means.right_mean, 0,
                        np.where(means.right_mean > means.left_mean, 1, np.nan))

    # free trial (e.g. trial 5)
    free = (g[g["trial"] == trial_idx]
            .merge(means["better"], left_on="game_number", right_index=True))

    choose_high = free["choice"] == free["better"]

    return {
        "prob_choose_high_mean":   choose_high.mean(),
        "avg_rt_high_mean_choice": free.loc[choose_high,  "RT"].mean(),
        "avg_rt_low_mean_choice":  free.loc[~choose_high, "RT"].mean()
    }

# ------------------------------------------------------------------
# This function iterates over a specified range (param_vals) for a specified  parameter (param_name), holding the other parameters constant (base_params).
# It performs n_runs simulations for each parameter value and computes the mean and standard error of the model-free metric function (metric_fn) for each parameter value.
# ------------------------------------------------------------------
def sweep(param_name: str,
          param_vals,
          base_params: dict,
          n_runs: int,
          metric_fn):
    out = []
    for v in param_vals:
        mvals = []
        for _ in range(n_runs):
            params = base_params.copy()   # keep original intact
            params[param_name] = v        # overwrite the swept key

            model = pyddm.gddm(
                drift = lambda drift_rwrd_diff_mod,drift_dcsn_noise_mod,drift_value,
                               sigma_d,sigma_r,baseline_noise,side_bias,directed_exp,
                               baseline_info_bonus,random_exp : drift_value,
                starting_position = lambda starting_position_value: starting_position_value,
                noise=1.0, bound="B", nondecision=0, T_dur=100,
                conditions=["game_number","gameLength","trial","r",
                            "drift_value","starting_position_value"],
                parameters=params,
                choice_names=("right","left")
            )
            sim_df = KF_DDM_model(social_media_sample, model, fit_or_sim="sim", sim_using_max_pdf=True)["data"]
            mvals.append(metric_fn(sim_df))

        out.append({param_name: v,
                    "mean_prob_choose_high_mean"     : np.mean([d["prob_choose_high_mean"] for d in mvals]),
                    "stderr_prob_choose_high_mean"   : sem(([d["prob_choose_high_mean"] for d in mvals])),
                    "avg_rt_high_obs_mean": np.mean([d["avg_rt_high_mean_choice"] for d in mvals]),
                    "stderr_avg_rt_high_obs_mean"   : sem(([d["avg_rt_high_mean_choice"] for d in mvals])),
                    "avg_rt_low_obs_mean" : np.mean([d["avg_rt_low_mean_choice"] for d in mvals]),
                    "stderr_avg_rt_low_obs_mean"   : sem(([d["avg_rt_low_mean_choice"] for d in mvals]))}),

        
    return pd.DataFrame(out)

# ------------------------------------------------------------------
# === USER SETUP ===================================================
# Specify the parameters to hold fixed during the parameter sweep. One of these parameters will
# be written over with the value of param_name.
base_params = dict(
    drift_rwrd_diff_mod  = .05,
    drift_dcsn_noise_mod = 1,
    B                    = 3,
    sigma_d              = 5,
    sigma_r              = 5,
    baseline_noise       = 0.05,
    side_bias            = 0,
    directed_exp         = 5,
    baseline_info_bonus  = -5,
    random_exp           = 3
)

param_name   = "random_exp"            # specify the parameter to sweep while holding others constant
param_vals   = np.linspace(1, 10, 10)            # set the range of parameters to sweep for the parameter param_name
n_runs       = 1                               # specify number of simulations to run for each set of parameters. Can leave at 1 if we are using the max pdf method (simulates a choice/rt based on the max pdf) instead of sampling from the distribution of choices/RTs.
game_len   = 9                               # specify the game length to use (5 or 9)
trial_idx  = 5                               # specify the trial index to use (5, 6, 7, etc.)
metric_fn    = lambda df: high_obs_mean(df, game_len=game_len, trial_idx=trial_idx)  # Use game_len to control whether plotting H1 (5) or H5 (9) games. Use trial_idx to control which which free choice to consider (5 = first free choice)
# ==================================================================
results = sweep(param_name, param_vals, base_params, n_runs, metric_fn)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
# Make subtitle
horizon_label = "Horizon 5" if game_len == 9 else "Horizon=1"
free_choice_label = f"Free Choice {trial_idx - 4}"  # since 5 => 1, 6 => 2, etc.
plot_context = f"{horizon_label}, {free_choice_label}"

# If every value in the stderr_prob_choose_high_mean column is NaN, skip the error bars. This happens when n_runs=1 since the standard error of a single value is NaN.
choose_high_observed_mean_standard_error = None if results["stderr_prob_choose_high_mean"].isna().all() else results["stderr_prob_choose_high_mean"]
plt.figure()
plt.errorbar(results[param_name], results["mean_prob_choose_high_mean"],
             yerr=choose_high_observed_mean_standard_error, marker="o", linestyle="-", capsize=4)
plt.xlabel(param_name)
plt.ylabel("P(choose higher observed mean)")
plt.title(f"{param_name} sweep: P(choose higher observed mean)")
plt.suptitle(plot_context, fontsize=10, y=0.95)
plt.grid(alpha=.3)
plt.tight_layout()
plt.show(block=False)


plt.figure()
# Handle missing stderr separately for both lines
avg_rt_high_obs_mean_standard_error = None if results["stderr_avg_rt_high_obs_mean"].isna().all() else results["stderr_avg_rt_high_obs_mean"]
avg_rt_low_obs_mean_standard_error  = None if results["stderr_avg_rt_low_obs_mean"].isna().all()  else results["stderr_avg_rt_low_obs_mean"]

plt.errorbar(results[param_name], results["avg_rt_high_obs_mean"],
             yerr=avg_rt_high_obs_mean_standard_error, label="High observed mean", marker="o", linestyle="-", capsize=4)

plt.errorbar(results[param_name], results["avg_rt_low_obs_mean"],
             yerr=avg_rt_low_obs_mean_standard_error, label="Low observed mean", marker="s", linestyle="--", capsize=4)

plt.xlabel(param_name)
plt.ylabel("Average RT")
plt.title(f"{param_name} sweep: RT for high vs low observed mean choices")
plt.suptitle(plot_context, fontsize=10, y=0.95)
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.show(block=False)

print("Hi")