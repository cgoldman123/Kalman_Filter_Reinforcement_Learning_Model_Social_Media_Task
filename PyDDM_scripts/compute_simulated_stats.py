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
# from scipy.stats import std

eps = np.finfo(float).eps


# ------------------------------------------------------------------
# Generic function to compute model-free info (e.g., mean reaction time across reward differences, prob choose high mean side) for the following conditions:
#   • game_len   – 5 for H1 games, 9 for H5 games, etc.
#   • trial_idx  – which free-choice trial to score (5, 6, 7, …)
# ------------------------------------------------------------------
def compute_stats_for_specific_horizon_and_choice(sim_data: pd.DataFrame,
                  game_len: int,
                  trial_idx: int) -> dict:

    # If game_len is a list and trial index is 5 (first free choice), iterate through each game length in the list
    if isinstance(game_len, list):
        results_dict = {}
        for game_length in game_len:
            # iterate through each game length in the list
            g = sim_data[sim_data["gameLength"] == game_length]

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
        g = sim_data[sim_data["gameLength"] == game_len]

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


def compute_stats_across_horizons_and_choices(sim_df) -> dict:
        sim_data = sim_df["data"]
        results_dict = {}
        for game_length in [5, 9]:
            # iterate through each game length in the list, filtering data to that game length
            g = sim_data[sim_data["gameLength"] == game_length]
            
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
            # choice_array = choice_array[choice_array > 4]
            choice_array.sort()
            # Step 2: Iterate through each choice number and compute mean RT
            for choice_number in choice_array:
                # Calculate mean RT and std for the specified choice number
                if choice_number > 4:  # Only consider free choices (trial > 4)
                    df_trial = g_complete[g_complete['trial'] == choice_number]                
                    mean_rt = df_trial['RT'].mean()
                    std_rt = df_trial['RT'].std()
                    mean_prob_choose_high_mean = (df_trial['choice'] == df_trial['better_side']).mean()
                    std_prob_choose_high_mean = (df_trial['choice'] == df_trial['better_side']).std()
                    mean_prob_choose_high_info = (df_trial['made_high_info_choice']).mean()
                    std_prob_choose_high_info = (df_trial['made_high_info_choice']).std()

                    results_dict[f"mean_rt_horizon_{game_length}_choice_{choice_number}"] = mean_rt
                    results_dict[f"std_rt_horizon_{game_length}_choice_{choice_number}"] = std_rt
                    results_dict[f"mean_prob_choose_high_mean_horizon_{game_length}_choice_{choice_number}"] = mean_prob_choose_high_mean
                    results_dict[f"std_prob_choose_high_mean_horizon_{game_length}_choice_{choice_number}"] = std_prob_choose_high_mean
                    results_dict[f"mean_prob_choose_high_info_horizon_{game_length}_choice_{choice_number}"] = mean_prob_choose_high_info
                    results_dict[f"std_prob_choose_high_info_horizon_{game_length}_choice_{choice_number}"] = std_prob_choose_high_info
                # For all choices, get the average total uncertainty, reward difference, and jensen shannon divergence for the option they chose
                # Create Total Uncertainty dataframe!!! 
                total_uncertainty_df = pd.DataFrame(sim_df["total_uncertainty"])
                # subset the dataframe to the game length and trial number
                horizon_mask = total_uncertainty_df.notna().sum(axis=1) == game_length
                filtered_total_uncertainty_df = total_uncertainty_df.loc[horizon_mask].iloc[:, choice_number - 1]
                results_dict[f"mean_tot_uncert_horizon_{game_length}_choice_{choice_number}"] = filtered_total_uncertainty_df.mean()
                results_dict[f"std_tot_uncert_horizon_{game_length}_choice_{choice_number}"] = filtered_total_uncertainty_df.std()

                # Create Mean Reward Difference dataframe!!! 
                reward_diff_df = pd.DataFrame(sim_df["rdiff_chosen_opt"])
                # subset the dataframe to the game length and trial number
                horizon_mask = reward_diff_df.notna().sum(axis=1) == game_length
                filtered_reward_rdiff_df = reward_diff_df.loc[horizon_mask].iloc[:, choice_number - 1]
                results_dict[f"mean_reward_diff_horizon_{game_length}_choice_{choice_number}"] = filtered_reward_rdiff_df.mean()
                results_dict[f"std_reward_diff_horizon_{game_length}_choice_{choice_number}"] = filtered_reward_rdiff_df.std()

                 # Create JSD dataframe!!! 
                jsd_df = pd.DataFrame(sim_df["jsd_diff_chosen_opt"])
                # subset the dataframe to the game length and trial number
                horizon_mask = jsd_df.notna().sum(axis=1) == game_length
                filtered_jsd_df = jsd_df.loc[horizon_mask].iloc[:, choice_number - 1]
                results_dict[f"mean_jsd_horizon_{game_length}_choice_{choice_number}"] = filtered_jsd_df.mean()
                results_dict[f"std_jsd_horizon_{game_length}_choice_{choice_number}"] = filtered_jsd_df.std()

                # Get total uncertainty, mean reward difference, and JSD for different generative reward differences
                for gen_diff in [2, 4, 8, 12, 24]:
                    # Find the game numbers where the generative reward difference is gen_diff and the horizon is game_length (since we are using g_complete which only contains the specified game length)
                    unique_games = g_complete.loc[abs(g_complete['reward_diff']) == gen_diff, 'game_number'].unique()
                    unique_games = unique_games - 1 # Adjust for zero-based indexing
                    # find the rows corresponding to these unique games
                    rows_in_unique_games = np.isin(np.arange(total_uncertainty_df.shape[0]), unique_games)
                    
                    # Get the total uncertainty for these rows
                    filtered_total_uncertainty_df_rdiff = total_uncertainty_df.loc[rows_in_unique_games].iloc[:, choice_number - 1]
                    results_dict[f"mean_tot_uncert_horizon_{game_length}_choice_{choice_number}_rdiff_{gen_diff}"] = filtered_total_uncertainty_df_rdiff.mean()
                    results_dict[f"std_tot_uncert_horizon_{game_length}_choice_{choice_number}_rdiff_{gen_diff}"] = filtered_total_uncertainty_df_rdiff.std()

                    filtered_reward_diff_df_rdiff = reward_diff_df.loc[rows_in_unique_games].iloc[:, choice_number - 1]
                    results_dict[f"mean_reward_diff_horizon_{game_length}_choice_{choice_number}_rdiff_{gen_diff}"] = filtered_reward_diff_df_rdiff.mean()
                    results_dict[f"std_reward_diff_horizon_{game_length}_choice_{choice_number}_rdiff_{gen_diff}"] = filtered_reward_diff_df_rdiff.std()

                    filtered_jsd_df_rdiff_rdiff = jsd_df.loc[rows_in_unique_games].iloc[:, choice_number - 1]
                    results_dict[f"mean_jsd_horizon_{game_length}_choice_{choice_number}_rdiff_{gen_diff}"] = filtered_jsd_df_rdiff_rdiff.mean()
                    results_dict[f"std_jsd_horizon_{game_length}_choice_{choice_number}_rdiff_{gen_diff}"] = filtered_jsd_df_rdiff_rdiff.std()



        return results_dict
            



# ------------------------------------------------------------------
# This function iterates over a specified range (param_vals) for a specified  parameter (param_name), holding the other parameters constant (base_params).
# It performs n_runs simulations for each parameter value and computes the mean and standard error of the model-free metric function (metric_fn) for each parameter value.
# ------------------------------------------------------------------
def stats_simulate_parameter_sweep(sample,
          settings,
          param_name: str,
          param_vals,
          base_params: dict,
          n_runs: int,
          metric_fn, sim_using_max_pdf, number_samples_to_sim):
    out = []
    for v in param_vals:
        mvals = []
        for _ in range(n_runs):
            params = base_params.copy()   # keep original intact
            params[param_name] = v        # overwrite the swept key

            model = pyddm.gddm(drift=lambda drift_dcsn_noise_mod,drift_value,sigma_d,sigma_r,baseline_noise,side_bias,directed_exp,baseline_info_bonus,random_exp,rel_uncert_mod, sigma_scaler : drift_value,
                          starting_position=lambda starting_position_value: starting_position_value, 
                          noise=1.0,     bound=lambda bound_intercept, bound_slope, t: max( bound_intercept + bound_slope*t, eps),  # linearly collapsing bound
                          nondecision=0, T_dur=7,
                          conditions=["game_number", "gameLength", "trial", "r", "drift_value","starting_position_value"],
                          parameters=params, choice_names=("right","left"))
            model.settings = settings
            sim_data = KF_DDM_model(sample, model, fit_or_sim="sim", sim_using_max_pdf=sim_using_max_pdf)["data"]
            mvals.append(metric_fn(sim_data))

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
                summary[f"stderr_{key}"] = np.std(values)
            # Append this summary to the output list
            out.append(summary)

        
    return pd.DataFrame(out)


# ------------------------------------------------------------------
# This function simulates a set of parameter values and computes the reaction time of the specified choice number for different reward differences
# ------------------------------------------------------------------
def stats_simulate_one_parameter_set(base_params: dict, game_len,trial_idx, settings, sample, sim_using_max_pdf, number_samples_to_sim):
    out = []
    model_free_across_horizons_and_choices = []
    model_free_for_specific_horizon_and_choice = []
    for _ in range(number_samples_to_sim):
        model = pyddm.gddm(drift=lambda drift_dcsn_noise_mod,drift_value,sigma_d,sigma_r,baseline_noise,side_bias,directed_exp,baseline_info_bonus,random_exp,rel_uncert_mod, sigma_scaler : drift_value,
                          starting_position=lambda starting_position_value: starting_position_value, 
                          noise=1.0,     bound=lambda bound_intercept, bound_slope, t: max( bound_intercept + bound_slope*t, eps),  # linearly collapsing bound
                          nondecision=0, T_dur=7,
                          conditions=["game_number", "gameLength", "trial", "r", "drift_value","starting_position_value"],
                          parameters=base_params, choice_names=("right","left"))
        model.settings = settings

        sim_df = KF_DDM_model(sample, model, fit_or_sim="sim", sim_using_max_pdf=sim_using_max_pdf)
        ### Compute statistics across horizons and choices ###
        model_free_across_horizons_and_choices.append(compute_stats_across_horizons_and_choices(sim_df))
        model_free_across_horizons_and_choices_df = pd.DataFrame(model_free_across_horizons_and_choices)

        sim_data = sim_df["data"]
        ### Compute statistics for the specific horizon and choice ###
        model_free_for_specific_horizon_and_choice.append(compute_stats_for_specific_horizon_and_choice(sim_data,game_len,trial_idx))
        
        # extract each RT-by-reward-difference DataFrame from mvals
        dfs = [d["avg_rt_by_reward_diff_game_len_5"] for d in model_free_for_specific_horizon_and_choice]
        # Combine 
        combined_stats_game_len_5 = pd.concat(dfs, axis=1)
        # Calculate mean and std across runs (row-wise)


        dfs = [d["avg_rt_by_reward_diff_game_len_9"] for d in model_free_for_specific_horizon_and_choice]
        # Combine 
        combined_stats_game_len_9 = pd.concat(dfs, axis=1)
        # Calculate mean and std across runs (row-wise)
    
    rt_by_reward_diff_summary_game_len_5 = pd.DataFrame({
        'reward_diff': combined_stats_game_len_5.index,
        'mean_RT': combined_stats_game_len_5.mean(axis=1),
        'std_RT': combined_stats_game_len_5.std(axis=1)
    })

    rt_by_reward_diff_summary_game_len_9 = pd.DataFrame({
        'reward_diff': combined_stats_game_len_9.index,
        'mean_RT': combined_stats_game_len_9.mean(axis=1),
        'std_RT': combined_stats_game_len_9.std(axis=1)
    })

    # If taking multiple samples, compute the mean of the means of each sample
    if len(model_free_across_horizons_and_choices_df) > 1:
        # Get relevant columns
        mean_cols = [col for col in model_free_across_horizons_and_choices_df.columns if col.startswith("mean_")]
        std_cols = [col for col in model_free_across_horizons_and_choices_df.columns if col.startswith("std_")]

        for mean_col, std_col in zip(mean_cols, std_cols):
            col_mean = model_free_across_horizons_and_choices_df[mean_col].mean()
            col_std  = model_free_across_horizons_and_choices_df[mean_col].std(ddof=1)

            # Assign that mean and std to the entire columns
            model_free_across_horizons_and_choices_df[mean_col] = col_mean
            model_free_across_horizons_and_choices_df[std_col]  = col_std
        # Take just one row since all rows are the same now
        model_free_across_horizons_and_choices_df = model_free_across_horizons_and_choices_df.iloc[[0]]


    return {
        "avg_rt_by_reward_diff_game_len_5": rt_by_reward_diff_summary_game_len_5,
        "avg_rt_by_reward_diff_game_len_9": rt_by_reward_diff_summary_game_len_9,
        "model_free_across_horizons_and_choices_df": model_free_across_horizons_and_choices_df
    }



