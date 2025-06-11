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

eps = np.finfo(float).eps

# Use this function to simulate model-free metrics under a range of parameter values

# Get arguments from the command line if they are passed in
if len(sys.argv) > 1:
    outpath_beh = sys.argv[1] # Behavioral file location
    results_dir = sys.argv[2] # Directory to save results
    id = sys.argv[3] # Subject ID
    room_type = sys.argv[4] # Room type (e.g., Like, Dislike)
    timestamp = sys.argv[5] # Timestamp (e.g., 04_16_25_T10-39-55)
    settings = sys.argv[6] # Settings for the model 
else:
    # Note you'll have to change both outpath_beh and id to fit another subject
    outpath_beh = f"L:/rsmith/lab-members/cgoldman/Wellbeing/social_media/output/SM_fits_PYDDM_test_PYDDM_04-21-2025_15-04-20/Like/model1/568d0641b5a2c2000cb657d0_beh_Like_04_21_25_T15-57-57.csv" # Behavioral file location
    id = "568d0641b5a2c2000cb657d0" # Subject ID
    results_dir = f"L:/rsmith/lab-members/cgoldman/Wellbeing/social_media/output/test/" # Directory to save results
    room_type = "Like" # Room type (e.g., Like, Dislike)
    timestamp = "current_timestamp" # Timestamp (e.g., 04_16_25_T10-39-55)
    settings = "Use_JSD_fit_all_RTs" # Settings for the model 

log_path = f"{results_dir}{id}_{room_type}_{timestamp}_model_log.txt"
sys.stdout = open(log_path, "w", buffering=1)  # line-buffered so that it updates the file in real-time
sys.stderr = sys.stdout  # Also capture errors
print("Welcome to the pyddm model simulating script!")

# Set the random seed for reproducibility
seed = 24
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
# Generic function to compute model-free info (e.g., mean reaction time across reward differences, prob choose high mean side) for the following conditions:
#   • game_len   – 5 for H1 games, 9 for H5 games, etc.
#   • trial_idx  – which free-choice trial to score (5, 6, 7, …)
# ------------------------------------------------------------------
def compute_stats_for_specific_horizon_and_choice(sim_df: pd.DataFrame,
                  game_len: int,
                  trial_idx: int) -> dict:

    # If game_len is a list and trial index is 5 (first free choice), iterate through each game length in the list
    if isinstance(game_len, list):
        results_dict = {}
        for game_length in game_len:
            # iterate through each game length in the list
            g = sim_df[sim_df["gameLength"] == game_length]

            # compute the means of the left and right options for the forced trials (1-4) 
            forced = g[g["trial"].isin([1, 2, 3, 4])]
            means = (forced
                    .groupby("game_number")
                    .agg(left_mean=('left_bandit_outcome',  'mean'),
                        right_mean=('right_bandit_outcome', 'mean')))
            # calculate the reward difference between the left and right options, rounding to nearest integer
            means["reward_diff"] = np.round(means.left_mean - means.right_mean,0)
            # label which option has the higher mean as 1 (right) or 0 (left)
            means["better_side"] = np.where(means.left_mean  > means.right_mean, 0,
                                np.where(means.right_mean > means.left_mean, 1, np.nan))
            
            # determine which side has more information in each game by summing the forced choices for that side 
            info_differences = (forced
                .groupby("game_number")
                .agg(num_right_forced_choices=('choice',  'sum')))
            
            # label the high information side as 1 (right) or 0 (left)
            info_differences["high_info_side"] = np.where(info_differences.num_right_forced_choices < 2, 1,
                                                    np.where(info_differences.num_right_forced_choices > 2, 0, np.nan))

            # filter to a free choice trial (e.g. trial 5) and merge with the means and info_differences dataframes
            free = (g[g["trial"] == trial_idx]
                    .merge(means[["better_side", "reward_diff"]], left_on="game_number", right_index=True)
                    .merge(info_differences[["high_info_side"]], left_on="game_number", right_index=True))

            # get the trials where the participant chose the option with the higher mean
            choose_high = free["choice"] == free["better_side"]
            choose_high_info = free["choice"] == free["high_info_side"]

            # Dynamically add keys with game_length in the name
            results_dict[f"prob_choose_high_mean_game_len_{game_length}"] = choose_high.mean()
            results_dict[f"prob_choose_high_info_game_len_{game_length}"] = choose_high_info.mean()
            results_dict[f"avg_rt_high_mean_choice_game_len_{game_length}"] = free.loc[choose_high,  "RT"].mean(),
            results_dict[f"avg_rt_low_mean_choice_game_len_{game_length}"]  = free.loc[~choose_high, "RT"].mean(),
            results_dict[f"avg_rt_by_reward_diff_game_len_{game_length}"]   = free.groupby("reward_diff")["RT"].mean()
            results_dict[f"avg_rt_game_len_{game_length}"]   = free["RT"].mean()    
            
        return results_dict
    else:
        # game length is not a list, so we can proceed subsetting to the specified game length
        g = sim_df[sim_df["gameLength"] == game_len]

        # compute the means of the left and right options for the forced trials (1-4) 
        forced = g[g["trial"].isin([1, 2, 3, 4])]
        means = (forced
                .groupby("game_number")
                .agg(left_mean=('left_bandit_outcome',  'mean'),
                    right_mean=('right_bandit_outcome', 'mean')))
        # calculate the reward difference between the left and right options, rounding to nearest integer
        means["reward_diff"] = np.round(means.left_mean - means.right_mean,0)
        # calculate which option has the higher mean
        means["better_side"] = np.where(means.left_mean  > means.right_mean, 0,
                            np.where(means.right_mean > means.left_mean, 1, np.nan))
        # free trial (e.g. trial 5)
        free = (g[g["trial"] == trial_idx]
                .merge(means[["better_side", "reward_diff"]], left_on="game_number", right_index=True))


        choose_high = free["choice"] == free["better_side"]

        return {
            "prob_choose_high_mean":   choose_high.mean(),
            "avg_rt_high_mean_choice": free.loc[choose_high,  "RT"].mean(),
            "avg_rt_low_mean_choice":  free.loc[~choose_high, "RT"].mean(),
            "avg_rt_by_reward_diff": free.groupby("reward_diff")["RT"].mean()
        }


def compute_stats_across_horizons_and_choices(sim_df: pd.DataFrame) -> dict:
        results_dict = {}
        for game_length in [5, 9]:
            # iterate through each game length in the list, filtering data to that game length
            g = sim_df[sim_df["gameLength"] == game_length]
            
            # compute the means of the left and right options for the forced trials (1-4) 
            forced = g[g["trial"].isin([1, 2, 3, 4])]
            means = (forced
                    .groupby("game_number")
                    .agg(left_mean=('left_bandit_outcome',  'mean'),
                        right_mean=('right_bandit_outcome', 'mean')))
            # calculate the reward difference between the left and right options, rounding to nearest integer
            means["reward_diff"] = np.round(means.left_mean - means.right_mean,0)
            # label which option has the higher mean as 1 (right) or 0 (left)
            means["better_side"] = np.where(means.left_mean  > means.right_mean, 0,
                                np.where(means.right_mean > means.left_mean, 1, np.nan))
            
            # Alternative way of calculating info difference based only on forced choices
            # info_differences = (forced
            #     .groupby("game_number")
            #     .agg(num_right_forced_choices=('choice',  'sum')))
            # g["made_high_info_choice"] = np.where(info_differences.num_right_forced_choices < 2, 1,
            #                             np.where(info_differences.num_right_forced_choices > 2, 0, np.nan))
                        # filter to a free choice trial (e.g. trial 5) and merge with the means and info_differences dataframes
            # g_complete = (g
            #         .merge(means[["better_side", "reward_diff"]], left_on="game_number", right_index=True)
            #         .merge(info_differences[["high_info_side"]], left_on="game_number", right_index=True))
            
            # Get the information difference based on the number of times the right side was chosen
            g = g.sort_values(by=['game_number', 'trial'])

            # Convert choice: 1 -> 1 (right), 0 -> -1 (left)
            choice_diff = g['choice'].replace(0, -1)
            # Compute cumulative sum per game, shifted so first trial starts at 0
            g['information_difference'] = (
                choice_diff
                .groupby(g['game_number'])
                .transform(lambda x: x.cumsum().shift(fill_value=0))
            )

            g['made_high_info_choice'] = np.where(
                g['information_difference'] > 0,  g['choice'] == 0,  # chose right when more info on right
                np.where(
                    g['information_difference'] < 0, g['choice'] == 1,  # chose left when more info on left
                    np.nan  # info difference is 0 → not defined
                )
            ).astype("float")  # Keep NaNs




            # filter to a free choice trial (e.g. trial 5) and merge with the means and info_differences dataframes
            g_complete = (g
                    .merge(means[["better_side", "reward_diff"]], left_on="game_number", right_index=True)
                    )
            
            # Step 1: Get free choice numbers (i.e., unique trial values > 4)
            choice_array = g_complete['trial'].unique()
            choice_array = choice_array[choice_array > 4]
            choice_array.sort()
            # Step 2: Iterate through each choice number and compute mean RT
            for choice_number in choice_array:
                # Calculate mean RT and SEM for the specified choice number

                df_trial = g_complete[g_complete['trial'] == choice_number]                
                mean_rt = df_trial['RT'].mean()
                sem_rt = df_trial['RT'].sem()
                mean_prob_choose_high_mean = (df_trial['choice'] == df_trial['better_side']).mean()
                sem_prob_choose_high_mean = (df_trial['choice'] == df_trial['better_side']).sem()
                mean_prob_choose_high_info = (df_trial['made_high_info_choice']).mean()
                sem_prob_choose_high_info = (df_trial['made_high_info_choice']).sem()

                results_dict[f"mean_rt_horizon_{game_length}_choice_{choice_number}"] = mean_rt
                results_dict[f"sem_rt_horizon_{game_length}_choice_{choice_number}"] = sem_rt
                results_dict[f"mean_prob_choose_high_mean_horizon_{game_length}_choice_{choice_number}"] = mean_prob_choose_high_mean
                results_dict[f"sem_prob_choose_high_mean_horizon_{game_length}_choice_{choice_number}"] = sem_prob_choose_high_mean
                results_dict[f"mean_prob_choose_high_info_horizon_{game_length}_choice_{choice_number}"] = mean_prob_choose_high_info
                results_dict[f"sem_prob_choose_high_info_horizon_{game_length}_choice_{choice_number}"] = sem_prob_choose_high_info


        
        
        return results_dict
            



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

            model = pyddm.gddm(drift=lambda drift_dcsn_noise_mod,drift_value,sigma_d,sigma_r,baseline_noise,side_bias,directed_exp,baseline_info_bonus,random_exp,rel_uncert_mod : drift_value,
                          starting_position=lambda starting_position_value: starting_position_value, 
                          noise=1.0,     bound=lambda bound_intercept, bound_slope, t: max( bound_intercept + bound_slope*t, eps),  # linearly collapsing bound
                          nondecision=0, T_dur=7,
                          conditions=["game_number", "gameLength", "trial", "r", "drift_value","starting_position_value"],
                          parameters=params, choice_names=("right","left"))
            model.settings = settings
            sim_df = KF_DDM_model(social_media_sample, model, fit_or_sim="sim", sim_using_max_pdf=True)["data"]
            mvals.append(metric_fn(sim_df))

            # Summarize the model free results in an output dictionary
            # Exclude keys you don't want to summarize
            exclude_keys = ["avg_rt_by_reward_diff_game_len_5", "avg_rt_by_reward_diff_game_len_9"]
            # Get all keys from the first result
            all_keys = mvals[0].keys()
            # Filter keys to keep only those you care about
            summary_keys = [k for k in all_keys if k not in exclude_keys]
            # Build summary dict dynamically
            summary = {param_name: v}
            for key in summary_keys:
                values = [d[key] for d in mvals]
                summary[f"mean_{key}"] = np.mean(values)
                summary[f"stderr_{key}"] = sem(values)
            # Append this summary to the output list
            out.append(summary)

        
    return pd.DataFrame(out)


# ------------------------------------------------------------------
# This function simulates a set of parameter values and computes the reaction time of the specified choice number for different reward differences
# ------------------------------------------------------------------
def get_rts_reward_difference(base_params: dict,
          n_runs: int, game_len,trial_idx, settings):
    out = []
    model_free_across_horizons_and_choices = []
    model_free_for_specific_horizon_and_choice = []
    for _ in range(n_runs):
        model = pyddm.gddm(drift=lambda drift_dcsn_noise_mod,drift_value,sigma_d,sigma_r,baseline_noise,side_bias,directed_exp,baseline_info_bonus,random_exp,rel_uncert_mod : drift_value,
                          starting_position=lambda starting_position_value: starting_position_value, 
                          noise=1.0,     bound=lambda bound_intercept, bound_slope, t: max( bound_intercept + bound_slope*t, eps),  # linearly collapsing bound
                          nondecision=0, T_dur=7,
                          conditions=["game_number", "gameLength", "trial", "r", "drift_value","starting_position_value"],
                          parameters=base_params, choice_names=("right","left"))
        model.settings = settings

        sim_df = KF_DDM_model(social_media_sample, model, fit_or_sim="sim", sim_using_max_pdf=True)["data"]
        ### Compute statistics across horizons and choices ###
        model_free_across_horizons_and_choices.append(compute_stats_across_horizons_and_choices(sim_df))
        model_free_across_horizons_and_choices_df = pd.DataFrame(model_free_across_horizons_and_choices)

        
        ### Compute statistics for the specific horizon and choice ###
        model_free_for_specific_horizon_and_choice.append(compute_stats_for_specific_horizon_and_choice(sim_df,game_len,trial_idx))
        
        # extract each RT-by-reward-difference DataFrame from mvals
        dfs = [d["avg_rt_by_reward_diff_game_len_5"] for d in model_free_for_specific_horizon_and_choice]
        # Combine 
        combined = pd.concat(dfs, axis=1)
        # Calculate mean and sem across runs (row-wise)
        rt_by_reward_diff_summary_game_len_5 = pd.DataFrame({
            'reward_diff': combined.index,
            'mean_RT': combined.mean(axis=1),
            'sem_RT': combined.sem(axis=1)
        })

        dfs = [d["avg_rt_by_reward_diff_game_len_9"] for d in model_free_for_specific_horizon_and_choice]
        # Combine 
        combined = pd.concat(dfs, axis=1)
        # Calculate mean and sem across runs (row-wise)
        rt_by_reward_diff_summary_game_len_9 = pd.DataFrame({
            'reward_diff': combined.index,
            'mean_RT': combined.mean(axis=1),
            'sem_RT': combined.sem(axis=1)
        })

    return {
        "avg_rt_by_reward_diff_game_len_5": rt_by_reward_diff_summary_game_len_5,
        "avg_rt_by_reward_diff_game_len_9": rt_by_reward_diff_summary_game_len_9,
        "model_free_across_horizons_and_choices_df": model_free_across_horizons_and_choices_df
    }




run_RT_by_reward_diff = True # adjust this to True if you want to run the RT by reward difference simulation
run_param_sweep = False # adjust this to True if you want to run the parameter sweep simulation


if run_RT_by_reward_diff:
    #Simulate RTs for a range of reward differences
    base_params = dict(
        drift_dcsn_noise_mod = 1,
        bound_intercept      = 2, #(1, 6)
        bound_slope          = -.1, #(-2, -.01)
        sigma_d              = 0, #(0,10)
        sigma_r              = 6, #(4,16)
        baseline_noise       = 1, #(.1,10)
        side_bias            = 0, #(-4,4)
        directed_exp         = 0.0, #(-4,4) #.15 reasonable value
        baseline_info_bonus  = -.07, #(-4,4) # -.07 reasonable value
        random_exp           = 1.7,  #(.1,10)
        rel_uncert_mod       = 0, #(-1,1)
    )

    n_runs       = 1                               # specify number of simulations to run for each set of parameters. Can leave at 1 if we are using the max pdf method (simulates a choice/rt based on the max pdf) instead of sampling from the distribution of choices/RTs.
    game_len   = [5, 9]                               # specify the game length to use ([5,9] )
    trial_idx  = 5                               # specify the trial index to use (5, 6, 7, etc.)
    # ==================================================================
    results = get_rts_reward_difference(base_params, n_runs, game_len,trial_idx,settings)

    plt.figure()
    summary_h1 = results['avg_rt_by_reward_diff_game_len_5']
    plt.errorbar(summary_h1['reward_diff'], summary_h1['mean_RT'], yerr=summary_h1['sem_RT'],
                 fmt='o-', color="blue",capsize=4, label="H1")
    summary_h5 = results['avg_rt_by_reward_diff_game_len_9']
    plt.errorbar(summary_h5['reward_diff'], summary_h5['mean_RT'], yerr=summary_h5['sem_RT'],
                 fmt='o-', color="red",capsize=4, label="H5")
    plt.xlabel('Reward Difference')
    plt.ylabel('Mean RT')
    plt.title('Mean RT by Reward Difference for the First Free Choice')
    plt.grid(True)
    plt.legend()  
    plt.show(block=False)


    # get mean RT for reward differences of absolute value 2 or 4 for each horizon
    h1_reward_diff_2_or_4 = summary_h1.loc[[2.0, -2.0, 4.0, -4.0], 'mean_RT'].mean()
    h5_reward_diff_2_or_4 = summary_h5.loc[[2.0, -2.0, 4.0, -4.0], 'mean_RT'].mean()
    h1_reward_diff_12_or_24 = summary_h1.loc[[12.0, -12.0, 24.0, -24.0], 'mean_RT'].mean()
    h5_reward_diff_12_or_24 = summary_h5.loc[[12.0, -12.0, 24.0, -24.0], 'mean_RT'].mean()

    print(f"Mean RT: {summary_h1['mean_RT'].mean()}")
    print(f"Mean RT: {summary_h5['mean_RT'].mean()}")


    ### Plot mean RT by choice number and horizon ###
    # Assume results['model_free_across_horizons_and_choices_df'] contains both mean and sem columns
    model_free_results = results['model_free_across_horizons_and_choices_df']
    # Melt mean RT
    df_mean = model_free_results.filter(like='mean_rt').melt(var_name="label", value_name="mean_rt")
    # Melt SEM
    df_sem = model_free_results.filter(like='sem_rt').melt(var_name="label", value_name="sem_rt")
    # Align by stripping 'sem_' and 'mean_' prefixes
    df_sem["label"] = df_sem["label"].str.replace("sem_", "mean_", regex=False)
    # Merge both long-format DataFrames
    df_long = pd.merge(df_mean, df_sem, on="label")
    # Extract horizon and choice number
    df_long["horizon"] = df_long["label"].str.extract(r"horizon_(\d+)").astype(int)
    df_long["choice_number"] = df_long["label"].str.extract(r"choice_(\d+)").astype(int)
    # Split into H1 and H5
    h5 = df_long[df_long["horizon"] == 5]
    h9 = df_long[df_long["horizon"] == 9]
    # Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(h9["choice_number"], h9["mean_rt"], yerr=h9["sem_rt"], fmt='o-', color="red", label="H5", capsize=4)
    plt.errorbar(h5["choice_number"], h5["mean_rt"], yerr=h5["sem_rt"], fmt='o', color="blue", label="H1", capsize=4)
    # Labels and formatting
    plt.xlabel("Choice Number")
    plt.ylabel("Mean Reaction Time (s)")
    plt.title("Mean RT by Choice Number and Horizon")
    plt.grid(True)
    plt.legend(title="Horizon")
    plt.tight_layout()
    plt.show(block=False)


    
    ### Plot prob choose the high mean option by choice number and horizon ###
    # Assume results['model_free_across_horizons_and_choices_df'] contains both mean and sem columns
    model_free_results = results['model_free_across_horizons_and_choices_df']
    # Melt mean RT
    df_mean = model_free_results.filter(like='mean_prob_choose_high_mean').melt(var_name="label", value_name="mean_prob_choose_high_mean")
    # Melt SEM
    df_sem = model_free_results.filter(like='sem_prob_choose_high_mean').melt(var_name="label", value_name="sem_prob_choose_high_mean")
    # Align by stripping 'sem_' and 'mean_' prefixes
    df_sem["label"] = df_sem["label"].str.replace("sem_", "mean_", regex=False)
    # Merge both long-format DataFrames
    df_long = pd.merge(df_mean, df_sem, on="label")
    # Extract horizon and choice number
    df_long["horizon"] = df_long["label"].str.extract(r"horizon_(\d+)").astype(int)
    df_long["choice_number"] = df_long["label"].str.extract(r"choice_(\d+)").astype(int)
    # Split into H1 and H5
    h5 = df_long[df_long["horizon"] == 5]
    h9 = df_long[df_long["horizon"] == 9]
    # Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(h9["choice_number"], h9["mean_prob_choose_high_mean"], yerr=h9["sem_prob_choose_high_mean"], fmt='o-', color="red", label="H5", capsize=4)
    plt.errorbar(h5["choice_number"], h5["mean_prob_choose_high_mean"], yerr=h5["sem_prob_choose_high_mean"], fmt='o', color="blue", label="H1", capsize=4)
    # Labels and formatting
    plt.xlabel("Choice Number")
    plt.ylabel("Prob Choose High Mean Option")
    plt.title("Prob Choose High Mean Option by Choice Number and Horizon")
    plt.grid(True)
    plt.legend(title="Horizon")
    plt.tight_layout()
    plt.show(block=False)


    ### Plot prob choose the high info option by choice number and horizon ###
    # Assume results['model_free_across_horizons_and_choices_df'] contains both mean and sem columns
    model_free_results = results['model_free_across_horizons_and_choices_df']
    # Melt mean RT
    df_mean = model_free_results.filter(like='mean_prob_choose_high_info').melt(var_name="label", value_name="mean_prob_choose_high_info")
    # Melt SEM
    df_sem = model_free_results.filter(like='sem_prob_choose_high_info').melt(var_name="label", value_name="sem_prob_choose_high_info")
    # Align by stripping 'sem_' and 'mean_' prefixes
    df_sem["label"] = df_sem["label"].str.replace("sem_", "mean_", regex=False)
    # Merge both long-format DataFrames
    df_long = pd.merge(df_mean, df_sem, on="label")
    # Extract horizon and choice number
    df_long["horizon"] = df_long["label"].str.extract(r"horizon_(\d+)").astype(int)
    df_long["choice_number"] = df_long["label"].str.extract(r"choice_(\d+)").astype(int)
    # Split into H1 and H5
    h5 = df_long[df_long["horizon"] == 5]
    h9 = df_long[df_long["horizon"] == 9]
    # Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(h9["choice_number"], h9["mean_prob_choose_high_info"], yerr=h9["sem_prob_choose_high_info"], fmt='o-', color="red", label="H5", capsize=4)
    plt.errorbar(h5["choice_number"], h5["mean_prob_choose_high_info"], yerr=h5["sem_prob_choose_high_info"], fmt='o', color="blue", label="H1", capsize=4)
    # Labels and formatting
    plt.xlabel("Choice Number")
    plt.ylabel("Prob Choose High Info Option")
    plt.title("Prob Choose High Info Option by Choice Number and Horizon")
    plt.grid(True)
    plt.legend(title="Horizon")
    plt.tight_layout()
    plt.show(block=False)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Get only the 4 most recent individual figures (skip the combined one)
    fig_nums = plt.get_fignums()[:4]

    for i, fignum in enumerate(fig_nums):
        fig_src = plt.figure(fignum)
        ax_src = fig_src.axes[0]
        fig_src.canvas.draw()
        buf = fig_src.canvas.buffer_rgba()
        axs[i // 2, i % 2].imshow(buf)
        axs[i // 2, i % 2].axis('off')

    plt.tight_layout()

    # Close only the individual figs (not the combined one)
    for fignum in fig_nums:
        plt.close(plt.figure(fignum))

    plt.show(block=False)



## PARAMETER SWEEP
# Specify the parameters to hold fixed during the parameter sweep. One of these parameters will
# be written over with the value of param_name.
if run_param_sweep:
    base_params = dict(
        drift_dcsn_noise_mod = 1,
        bound_intercept      = 2, #(1, 6)
        bound_slope          = -.1, #(-2, -.01)
        sigma_d              = 0, #(0,10)
        sigma_r              = 6, #(4,16)
        baseline_noise       = 1, #(.1,10)
        side_bias            = 0, #(-4,4)
        directed_exp         = 0.18, #(-4,4) #.15 reasonable value
        baseline_info_bonus  = -.07, #(-4,4) # -.07 reasonable value
        random_exp           = 1.7,  #(.1,10)
        rel_uncert_mod       = .2, #(-1,1)
    )

    param_name   = "sigma_r"            # specify the parameter to sweep while holding others constant
    param_vals   = np.linspace(4, 16, 20)            # set the range of parameters to sweep for the parameter param_name
    n_runs       = 1                               # specify number of simulations to run for each set of parameters. Can leave at 1 if we are using the max pdf method (simulates a choice/rt based on the max pdf) instead of sampling from the distribution of choices/RTs.
    game_len   =  [5,9] #[5,9] or [9]                              # specify the game length; use brackets
    trial_idx  = 5                               # specify the trial index to use (5, 6, 7, etc.)
    metric_fn    = lambda df: compute_stats_for_specific_horizon_and_choice(df, game_len=game_len, trial_idx=trial_idx)  # Use game_len to control whether plotting H1 (5) or H5 (9) games. Use trial_idx to control which which free choice to consider (5 = first free choice)
    # ==================================================================
    results = sweep(param_name, param_vals, base_params, n_runs, metric_fn)

    # ------------------------------------------------------------------
    # Plot: Probability of choosing the option with the higher observed mean for each horizon across the parameter sweep.
    # ------------------------------------------------------------------
    # Make subtitle
    free_choice_label = f"Free Choice {trial_idx - 4}"  # since 5 => 1, 6 => 2, etc.
    
    plt.figure()
    # If every value in the stderr_prob_choose_high_mean column is NaN, skip the error bars. This happens when n_runs=1 since the standard error of a single value is NaN.
    if 5 in game_len:
        choose_high_observed_mean_h1_standard_error = None if results["stderr_prob_choose_high_mean_game_len_5"].isna().all() else results["stderr_prob_choose_high_mean_game_len_5"]
        plt.errorbar(results[param_name], results["mean_prob_choose_high_mean_game_len_5"],
                yerr=choose_high_observed_mean_h1_standard_error, marker="o", linestyle="-", capsize=4, label="H1")
    if 9 in game_len:
        choose_high_observed_mean_h5_standard_error = None if results["stderr_prob_choose_high_mean_game_len_9"].isna().all() else results["stderr_prob_choose_high_mean_game_len_9"] 
        plt.errorbar(results[param_name], results["mean_prob_choose_high_mean_game_len_9"],
                    yerr=choose_high_observed_mean_h5_standard_error, marker="o", linestyle="-", capsize=4, label="H5")
    plt.xlabel(param_name)
    plt.ylabel("P(choose higher observed mean)")
    plt.title(f"{param_name} sweep: P(choose higher observed mean)")
    plt.suptitle(free_choice_label, fontsize=10, y=0.95)
    plt.grid(alpha=.3)
    plt.legend()  
    plt.tight_layout()
    plt.show(block=False)


    # ------------------------------------------------------------------
    # Plot: Probability of choosing the option with the higher info side for each horizon across the parameter sweep.
    # ------------------------------------------------------------------
    # Make subtitle
    free_choice_label = f"Free Choice {trial_idx - 4}"  # since 5 => 1, 6 => 2, etc.

    plt.figure()
    # If every value in the stderr_prob_choose_high_mean column is NaN, skip the error bars. This happens when n_runs=1 since the standard error of a single value is NaN.
    if 5 in game_len:
        choose_high_info_h1_standard_error = None if results["stderr_prob_choose_high_info_game_len_5"].isna().all() else results["stderr_prob_choose_high_info_game_len_5"]
        plt.errorbar(results[param_name], results["mean_prob_choose_high_info_game_len_5"],
                    yerr=choose_high_info_h1_standard_error, marker="o", linestyle="-", capsize=4, label="H1")
    if 9 in game_len:
        choose_high_info_h5_standard_error = None if results["stderr_prob_choose_high_info_game_len_9"].isna().all() else results["stderr_prob_choose_high_info_game_len_9"]
        plt.errorbar(results[param_name], results["mean_prob_choose_high_info_game_len_9"],
                    yerr=choose_high_info_h5_standard_error, marker="o", linestyle="-", capsize=4, label="H5")
    plt.xlabel(param_name)
    plt.ylabel("P(choose higher info side)")
    plt.title(f"{param_name} sweep: P(choose higher info side)")
    plt.suptitle(free_choice_label, fontsize=10, y=0.95)
    plt.grid(alpha=.3)
    plt.legend()  
    plt.tight_layout()
    plt.show(block=False)



    # ------------------------------------------------------------------
    # Plot: Average RT for each horizon across the parameter sweep. Note I could also separate by RTs for high and low observed means.
    # ------------------------------------------------------------------
    plt.figure()
    # Handle missing stderr separately for both lines
    if 5 in game_len:
        avg_rt_h1_standard_error = None if results["stderr_avg_rt_game_len_5"].isna().all() else results["stderr_avg_rt_game_len_5"]
        plt.errorbar(results[param_name], results["mean_avg_rt_game_len_5"],
                yerr=avg_rt_h1_standard_error, label="H1", marker="o", linestyle="-", capsize=4)
    if 9 in game_len:
        avg_rt_h5_standard_error  = None if results["stderr_avg_rt_game_len_9"].isna().all()  else results["stderr_avg_rt_game_len_9"]
        plt.errorbar(results[param_name], results["mean_avg_rt_game_len_9"],
                    yerr=avg_rt_h5_standard_error, label="H5", marker="s", linestyle="--", capsize=4)
    plt.xlabel(param_name)
    plt.ylabel("Average RT")
    plt.title(f"{param_name} sweep: RT")
    plt.suptitle(free_choice_label, fontsize=10, y=0.95)
    plt.legend()
    plt.grid(alpha=.3)
    plt.legend()  
    plt.tight_layout()
    plt.show(block=False)




print("Hi")

