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
from compute_simulated_stats import *
# from scipy.stats import std

eps = np.finfo(float).eps

# Note you'll have to change both outpath_beh and id to fit another subject
outpath_beh = f"L:/rsmith/lab-members/cgoldman/Wellbeing/social_media/output/SM_fits_PYDDM_test_PYDDM_04-21-2025_15-04-20/Like/model1/568d0641b5a2c2000cb657d0_beh_Like_04_21_25_T15-57-57.csv" # Behavioral file location
id = "568d0641b5a2c2000cb657d0" # Subject ID
results_dir = f"L:/rsmith/lab-members/cgoldman/Wellbeing/social_media/output/test/" # Directory to save results
room_type = "Like" # Room type (e.g., Like, Dislike)
timestamp = "current_timestamp" # Timestamp (e.g., 04_16_25_T10-39-55)
# settings = "" # Settings for the model Use_z_score_fit_all_RTs, Use_JSD_fit_all_RTs
settings = dict()

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

########### SETTINGS ##############

run_one_param_set = True # adjust this to True if you want to run the RT by reward difference simulation
run_param_sweep = True # adjust this to True if you want to run the parameter sweep simulation
sim_using_max_pdf = True # If True, the model will simulate a choice/RT based on the maximum of the simulated pdf. If False, it will sample from the distribution of choices/RTs.
plot_latent_states_separated_by_rdiff = True # If True, the latent states will be plotted separately for each reward difference. If False, they will be averaged across all reward differences.
plot_jsd = False # If True, the JSD will be plotted in the latent states plot. If False, it will not be plotted.
if not sim_using_max_pdf:
    number_samples_to_sim = 5
else:
    number_samples_to_sim = 1

base_params = dict(
    rdiff_bias_mod = -.05,
    random_exp = 5,
    bound_intercept = 3, #2.5 works
    base_noise = .0833,
    cong_DE = -3,
    incong_DE = -4,
    cong_base_info_bonus = 1,
    incong_base_info_bonus = 2,
    sigma_d = 6,
    sigma_r = 8,
    side_bias = 0,
    nondecision_time = 0.05,
)
 
 
 

###############################

if run_param_sweep:
    param_name   = "sigma_r"            # specify the parameter to sweep while holding others constant
    param_vals   = np.linspace(4, 16, 3)            # set the range of parameters to sweep for the parameter param_name
    trial_idx  = 5 # Choice number (5 to 9)


settings["plot_jsd"] = plot_jsd # Set the settings for the model to plot JSD or not


if run_one_param_set:
    game_lens   = [5, 9]                               # specify the game length to use in the reward difference plots ([5,9] or [9])
    trial_indices  = [5,6,7,8,9]                          # specify the trial indices to plot [5,6,7,8,9] or [5] etc.
    plot_error_bars = False
    # ==================================================================
    results = stats_simulate_one_parameter_set(base_params, game_lens,settings, social_media_sample, sim_using_max_pdf, number_samples_to_sim)

    ### Plot mean RT by reward difference and horizon ###
    plt.figure()

    for game_len in game_lens:
        for trial_idx in trial_indices:
            if game_len == 9 or (game_len == 5 and trial_idx == 5):
                summary = results['rt_by_reward_diff_summary'][f'game_len_{game_len}_choice_{trial_idx}']
                if plot_error_bars:
                    plt.errorbar(summary['reward_diff'], summary['mean_RT_reward_diff'], yerr=summary['std_RT_reward_diff'],
                                fmt='o-',capsize=4, label=f"H: {game_len-4}, Choice: {trial_idx}")
                else:
                    plt.plot(summary['reward_diff'], summary['mean_RT_reward_diff'],
                     marker='o', linestyle='-', label=f"H: {game_len-4}, Choice: {trial_idx}")

    plt.xlabel('Reward Difference')
    plt.ylabel('Mean RT')
    plt.title(f'Mean RT by Reward Difference')
    plt.grid(True)
    plt.legend()  
    plt.show(block=False)



    ### Plot mean RT by reward difference for high - low info options and horizon ###
    plt.figure()
    for game_len in game_lens:
        for trial_idx in trial_indices:
            if game_len == 9 or (game_len == 5 and trial_idx == 5):
                summary = results['rt_by_reward_diff_summary'][f'game_len_{game_len}_choice_{trial_idx}']
                if plot_error_bars:
                    plt.errorbar(summary['reward_diff'], summary['mean_RT_reward_diff_high_info_minus_low_info'], yerr=summary['std_RT_reward_diff_high_info_minus_low_info'],
                                fmt='o-',capsize=4, label=f"H: {game_len-4}, Choice: {trial_idx}")
                else:
                    plt.plot(summary['reward_diff'], summary['mean_RT_reward_diff_high_info_minus_low_info'],
                     marker='o', linestyle='-', label=f"H: {game_len-4}, Choice: {trial_idx}")

    plt.xlabel('Reward Difference')
    plt.ylabel('Mean RT')
    plt.title(f'Mean RT by Reward Difference for high - low info option ')
    plt.grid(True)
    plt.legend()  
    plt.show(block=False)
    # get mean RT for reward differences of absolute value 2 or 4 for each horizon
    # h1_reward_diff_2_or_4 = summary_h1.loc[[2.0, -2.0, 4.0, -4.0], 'mean_RT_reward_diff_high_info_minus_low_info'].mean()
    # h5_reward_diff_2_or_4 = summary_h5.loc[[2.0, -2.0, 4.0, -4.0], 'mean_RT_reward_diff_high_info_minus_low_info'].mean()
    # h1_reward_diff_12_or_24 = summary_h1.loc[[12.0, -12.0, 24.0, -24.0], 'mean_RT_reward_diff_high_info_minus_low_info'].mean()
    # h5_reward_diff_12_or_24 = summary_h5.loc[[12.0, -12.0, 24.0, -24.0], 'mean_RT_reward_diff_high_info_minus_low_info'].mean()





    ### Plot mean RT by choice number and horizon ###
    # Assume results['model_free_across_horizons_and_choices_df'] contains both mean and std columns
    model_free_results = results['model_free_across_horizons_and_choices_df']
    # Filter to RT cols and melt
    cols = [col for col in model_free_results.columns if "mean_rt" in col and "choice" in col]
    df_mean = model_free_results[cols].melt(var_name="label", value_name="mean_rt")
    # Melt std
    cols = [col for col in model_free_results.columns if "std_rt" in col and "choice" in col]
    df_std = model_free_results[cols].melt(var_name="label", value_name="std_rt")
    # Align by stripping 'std_' and 'mean_' prefixes
    df_std["label"] = df_std["label"].str.replace("std_", "mean_", regex=False)
    # Merge both long-format DataFrames
    df_long = pd.merge(df_mean, df_std, on="label")
    # Extract horizon and choice number
    df_long["horizon"] = df_long["label"].str.extract(r"horizon_(\d+)").astype(int)
    df_long["choice_number"] = df_long["label"].str.extract(r"choice_(\d+)").astype(int)
    # Split into H1 and H5
    h5 = df_long[df_long["horizon"] == 5]
    h9 = df_long[df_long["horizon"] == 9]
    # Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(h9["choice_number"], h9["mean_rt"], yerr=h9["std_rt"], fmt='o-', color="red", label="H5", capsize=4)
    plt.errorbar(h5["choice_number"], h5["mean_rt"], yerr=h5["std_rt"], fmt='o', color="blue", label="H1", capsize=4)
    # Labels and formatting
    plt.xlabel("Choice Number")
    plt.ylabel("Mean Reaction Time (s)")
    plt.title("Mean RT by Choice Number and Horizon")
    plt.grid(True)
    plt.legend(title="Horizon")
    plt.tight_layout()
    plt.show(block=False)


    
    ### Plot prob choose the high mean option by choice number and horizon ###
    # Assume results['model_free_across_horizons_and_choices_df'] contains both mean and std columns
    model_free_results = results['model_free_across_horizons_and_choices_df']
    # Melt mean RT
    cols = [col for col in model_free_results.columns if "mean_prob_choose_high_mean" in col and "choice" in col]
    df_mean = model_free_results[cols].melt(var_name="label", value_name="mean_prob_choose_high_mean")
    # Melt std
    cols = [col for col in model_free_results.columns if "std_prob_choose_high_mean" in col and "choice" in col]
    df_std = model_free_results[cols].melt(var_name="label", value_name="std_prob_choose_high_mean")

    # Align by stripping 'std_' and 'mean_' prefixes
    df_std["label"] = df_std["label"].str.replace("std_", "mean_", regex=False)
    # Merge both long-format DataFrames
    df_long = pd.merge(df_mean, df_std, on="label")
    # Extract horizon and choice number
    df_long["horizon"] = df_long["label"].str.extract(r"horizon_(\d+)").astype(int)
    df_long["choice_number"] = df_long["label"].str.extract(r"choice_(\d+)").astype(int)
    # Split into H1 and H5
    h5 = df_long[df_long["horizon"] == 5]
    h9 = df_long[df_long["horizon"] == 9]
    # Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(h9["choice_number"], h9["mean_prob_choose_high_mean"], yerr=h9["std_prob_choose_high_mean"], fmt='o-', color="red", label="H5", capsize=4)
    plt.errorbar(h5["choice_number"], h5["mean_prob_choose_high_mean"], yerr=h5["std_prob_choose_high_mean"], fmt='o', color="blue", label="H1", capsize=4)
    # Labels and formatting
    plt.xlabel("Choice Number")
    plt.ylabel("Prob Choose High Mean Option")
    plt.title("Prob Choose High Mean Option by Choice Number and Horizon")
    plt.grid(True)
    plt.legend(title="Horizon")
    plt.tight_layout()
    plt.show(block=False)


    ### Plot prob choose the high info option by choice number and horizon ###
    # Assume results['model_free_across_horizons_and_choices_df'] contains both mean and std columns
    model_free_results = results['model_free_across_horizons_and_choices_df']
    # Melt mean RT
    cols = [col for col in model_free_results.columns if "mean_prob_choose_high_info" in col and "choice" in col]
    df_mean = model_free_results[cols].melt(var_name="label", value_name="mean_prob_choose_high_info")
    # Melt std
    cols = [col for col in model_free_results.columns if "std_prob_choose_high_info" in col and "choice" in col]
    df_std = model_free_results[cols].melt(var_name="label", value_name="std_prob_choose_high_info")
    # Align by stripping 'std_' and 'mean_' prefixes
    df_std["label"] = df_std["label"].str.replace("std_", "mean_", regex=False)
    # Merge both long-format DataFrames
    df_long = pd.merge(df_mean, df_std, on="label")
    # Extract horizon and choice number
    df_long["horizon"] = df_long["label"].str.extract(r"horizon_(\d+)").astype(int)
    df_long["choice_number"] = df_long["label"].str.extract(r"choice_(\d+)").astype(int)
    # Split into H1 and H5
    h5 = df_long[df_long["horizon"] == 5]
    h9 = df_long[df_long["horizon"] == 9]
    # Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(h9["choice_number"], h9["mean_prob_choose_high_info"], yerr=h9["std_prob_choose_high_info"], fmt='o-', color="red", label="H5", capsize=4)
    plt.errorbar(h5["choice_number"], h5["mean_prob_choose_high_info"], yerr=h5["std_prob_choose_high_info"], fmt='o', color="blue", label="H1", capsize=4)
    # Labels and formatting
    plt.xlabel("Choice Number")
    plt.ylabel("Prob Choose High Info Option")
    plt.title("Prob Choose High Info Option by Choice Number and Horizon")
    plt.grid(True)
    plt.legend(title="Horizon")
    plt.tight_layout()
    plt.show(block=False)

    # Create one big plot with the individual figures combined into a 2x2 grid.
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

    plt.tight_layout(rect=[0, 0, 1, 0.92])  # leave space for title + subtitle

    # Add title and conditional subtitle
    fig.suptitle("Simulated behavior using one parameter set", fontsize=16)

    if sim_using_max_pdf:
        subtitle = "Simulations were based on the maximum of the pdf of the choice×RT distribution."
    else:
        subtitle = f"Simulations were sampled from the pdf of the choice×RT distribution {number_samples_to_sim} times."

    fig.text(0.5, 0.935, subtitle, ha='center', fontsize=10)

    # Close only the individual figs (not the combined one)
    for fignum in fig_nums:
        plt.close(plt.figure(fignum))

    plt.show(block=False)





    ###### PLOT LATENT STATES ######


    ### Plot total uncertainty by choice number and horizon ###
    model_free_results = results['model_free_across_horizons_and_choices_df']

    # Melt mean and std columns to long format
    melted_df_mean = model_free_results.filter(like='mean_tot_uncert').melt(var_name="label", value_name="mean_tot_uncert")
    melted_df_std = model_free_results.filter(like='std_tot_uncert').melt(var_name="label", value_name="std_tot_uncert")

    # Separate by whether labels contain specific reward differences
    df_mean_foreach_rdiff = melted_df_mean[melted_df_mean['label'].str.contains('rdiff_')].copy()
    df_mean_averaged_across_rdiff = melted_df_mean[~melted_df_mean['label'].str.contains('rdiff_')].copy()
    df_std_foreach_rdiff = melted_df_std[melted_df_std['label'].str.contains('rdiff_')].copy()
    df_std_averaged_across_rdiff = melted_df_std[~melted_df_std['label'].str.contains('rdiff_')].copy()

    # Choose which version to use
    if plot_latent_states_separated_by_rdiff:
        df_mean = df_mean_foreach_rdiff.copy()
        df_std = df_std_foreach_rdiff.copy()
    else:
        df_mean = df_mean_averaged_across_rdiff.copy()
        df_std = df_std_averaged_across_rdiff.copy()

    # Align std and mean by matching label names
    df_std["label"] = df_std["label"].str.replace("std_", "mean_", regex=False)
    df_long = pd.merge(df_mean, df_std, on="label")

    # Extract horizon and choice number from label strings
    df_long["horizon"] = df_long["label"].str.extract(r"horizon_(\d+)")[0].astype(int)
    df_long["choice_number"] = df_long["label"].str.extract(r"choice_(\d+)")[0].astype(int)

    plt.figure(figsize=(8, 5))

    if plot_latent_states_separated_by_rdiff:
        # Extract rdiff values
        df_long["rdiff"] = df_long["label"].str.extract(r"rdiff_(\d+)")[0].astype(int)

        rdiff_values = sorted(df_long["rdiff"].unique())
        colors = plt.cm.tab10.colors  # Up to 10 distinct colors

        for i, rdiff_val in enumerate(rdiff_values):
            for horizon_val, linestyle in zip([5, 9], [':', '-']):  # dotted for H5, solid for H9
                subset = df_long[(df_long["rdiff"] == rdiff_val) & (df_long["horizon"] == horizon_val)]
                plt.plot(subset["choice_number"], subset["mean_tot_uncert"],
                        marker='o', linestyle=linestyle, color=colors[i % len(colors)],
                        label=f"r_diff = {rdiff_val}, H{horizon_val}")

        plt.legend(title="Reward Difference")
    else:
        # Subset by horizon and plot with std error bars
        h5 = df_long[df_long["horizon"] == 5]
        h9 = df_long[df_long["horizon"] == 9]

        plt.errorbar(h9["choice_number"], h9["mean_tot_uncert"], yerr=h9["std_tot_uncert"],
                    fmt='o-', color="red", label="H5", capsize=4)
        plt.errorbar(h5["choice_number"], h5["mean_tot_uncert"], yerr=h5["std_tot_uncert"],
                    fmt='o:', color="blue", label="H1", capsize=4)

        plt.legend(title="Horizon")

    plt.xlabel("Choice Number")
    plt.ylabel("Total Uncertainty")
    plt.title("Total Uncertainty by Choice Number")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)


    ### Reward difference by choice number and horizon ###
    model_free_results = results['model_free_across_horizons_and_choices_df']

    # Melt wide-format mean and std columns to long format
    melted_df_mean = model_free_results.filter(like='mean_reward_diff').melt(var_name="label", value_name="mean_reward_diff")
    melted_df_std = model_free_results.filter(like='std_reward_diff').melt(var_name="label", value_name="std_reward_diff")

    # Separate rows that include a specific reward difference (rdiff) vs. those averaged across all
    df_mean_foreach_rdiff = melted_df_mean[melted_df_mean['label'].str.contains('rdiff_')].copy()
    df_mean_averaged_across_rdiff = melted_df_mean[~melted_df_mean['label'].str.contains('rdiff_')].copy()
    df_std_foreach_rdiff = melted_df_std[melted_df_std['label'].str.contains('rdiff_')].copy()
    df_std_averaged_across_rdiff = melted_df_std[~melted_df_std['label'].str.contains('rdiff_')].copy()

    # Choose which version to use based on plotting flag
    if plot_latent_states_separated_by_rdiff:
        df_mean = df_mean_foreach_rdiff.copy()
        df_std = df_std_foreach_rdiff.copy()
    else:
        df_mean = df_mean_averaged_across_rdiff.copy()
        df_std = df_std_averaged_across_rdiff.copy()

    # Align std and mean by matching label names
    df_std["label"] = df_std["label"].str.replace("std_", "mean_", regex=False)
    df_long = pd.merge(df_mean, df_std, on="label")

    # Extract horizon and choice number from label strings
    df_long["horizon"] = df_long["label"].str.extract(r"horizon_(\d+)")[0].astype(int)
    df_long["choice_number"] = df_long["label"].str.extract(r"choice_(\d+)")[0].astype(int)

    plt.figure(figsize=(8, 5))

    if plot_latent_states_separated_by_rdiff:
        # Extract rdiff values
        df_long["rdiff"] = df_long["label"].str.extract(r"rdiff_(\d+)")[0].astype(int)

        # Get unique rdiffs and assign colors
        rdiff_values = sorted(df_long["rdiff"].unique())
        colors = plt.cm.tab10.colors  # Up to 10 distinct colors

        for i, rdiff_val in enumerate(rdiff_values):
            for horizon_val, linestyle in zip([5, 9], [':', '-']):  # dotted for 5, solid for 9
                subset = df_long[(df_long["rdiff"] == rdiff_val) & (df_long["horizon"] == horizon_val)]
                plt.plot(subset["choice_number"], subset["mean_reward_diff"],
                        marker='o', linestyle=linestyle, color=colors[i % len(colors)],
                        label=f"r_diff = {rdiff_val}, H{horizon_val}")

        plt.legend(title="Reward Difference")
    else:
        # Subset by horizon and plot with std error bars
        h5 = df_long[df_long["horizon"] == 5]
        h9 = df_long[df_long["horizon"] == 9]

        plt.errorbar(h9["choice_number"], h9["mean_reward_diff"], yerr=h9["std_reward_diff"],
                    fmt='o-', color="red", label="H5", capsize=4)
        plt.errorbar(h5["choice_number"], h5["mean_reward_diff"], yerr=h5["std_reward_diff"],
                    fmt='o', color="blue", label="H1", capsize=4)

        plt.legend(title="Horizon")

    # Add axis labels and format the plot
    plt.xlabel("Choice Number")
    plt.ylabel("Reward Difference")
    plt.title("Reward Difference by Choice Number and Horizon")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)



    ### Plot JSD by choice number and horizon ###
    if plot_jsd:
        model_free_results = results['model_free_across_horizons_and_choices_df']

        # Melt wide-format mean and std columns to long format
        df_mean = model_free_results.filter(like='mean_jsd').melt(var_name="label", value_name="mean_jsd")
        df_std = model_free_results.filter(like='std_jsd').melt(var_name="label", value_name="std_jsd")

        # Separate rows that include a specific reward difference (rdiff) vs. those averaged across all
        df_mean_foreach_rdiff = df_mean[df_mean['label'].str.contains('rdiff_')].copy()
        df_mean_averaged_across_rdiff = df_mean[~df_mean['label'].str.contains('rdiff_')].copy()
        df_std_foreach_rdiff = df_std[df_std['label'].str.contains('rdiff_')].copy()
        df_std_averaged_across_rdiff = df_std[~df_std['label'].str.contains('rdiff_')].copy()

        # Choose which version to use based on plotting flag
        if plot_latent_states_separated_by_rdiff:
            df_mean = df_mean_foreach_rdiff.copy()
            df_std = df_std_foreach_rdiff.copy()
        else:
            df_mean = df_mean_averaged_across_rdiff.copy()
            df_std = df_std_averaged_across_rdiff.copy()

        # Align std and mean by matching label names
        df_std["label"] = df_std["label"].str.replace("std_", "mean_", regex=False)
        df_long = pd.merge(df_mean, df_std, on="label")

        # Extract horizon and choice number from label strings
        df_long["horizon"] = df_long["label"].str.extract(r"horizon_(\d+)")[0].astype(int)
        df_long["choice_number"] = df_long["label"].str.extract(r"choice_(\d+)")[0].astype(int)

        plt.figure(figsize=(8, 5))

        if plot_latent_states_separated_by_rdiff:
            # Extract rdiff values
            df_long["rdiff"] = df_long["label"].str.extract(r"rdiff_(\d+)")[0].astype(int)

            rdiff_values = sorted(df_long["rdiff"].unique())
            colors = plt.cm.tab10.colors  # Up to 10 distinct colors

            for i, rdiff_val in enumerate(rdiff_values):
                for horizon_val, linestyle in zip([5, 9], [':', '-']):
                    subset = df_long[(df_long["rdiff"] == rdiff_val) & (df_long["horizon"] == horizon_val)]
                    plt.plot(subset["choice_number"], subset["mean_jsd"],
                            marker='o', linestyle=linestyle, color=colors[i % len(colors)],
                            label=f"r_diff = {rdiff_val}, H{horizon_val}")

            plt.legend(title="Reward Difference")
        else:
            h5 = df_long[df_long["horizon"] == 5]
            h9 = df_long[df_long["horizon"] == 9]

            plt.errorbar(h9["choice_number"], h9["mean_jsd"], yerr=h9["std_jsd"],
                        fmt='o-', color="red", label="H5", capsize=4)
            plt.errorbar(h5["choice_number"], h5["mean_jsd"], yerr=h5["std_jsd"],
                        fmt='o:', color="blue", label="H1", capsize=4)

            plt.legend(title="Horizon")

        # Labels and formatting
        plt.xlabel("Choice Number")
        plt.ylabel("JSD")
        plt.title("JSD by Choice Number and Horizon")
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)



    
    # ------------------------------------------------------------------
    # Combine the last 3 single-panel figures into one multipanel fig
    # ------------------------------------------------------------------
    from matplotlib.gridspec import GridSpec

    if plot_jsd:
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 2, figure=fig)
        ax0 = fig.add_subplot(gs[0, :])      # JSD (top row, full width)
        ax1 = fig.add_subplot(gs[1, 0])      # Reward diff
        ax2 = fig.add_subplot(gs[1, 1])      # Total uncertainty

        # Grab the three most-recent model-free plots: reward diff, uncertainty, JSD
        fig_nums = plt.get_fignums()[-4:]  # Include the one before the model-free plots
        fig_nums = fig_nums[-3:]           # Keep only the last three
        fig_nums = [x - 1 for x in fig_nums] # Adjust for 0-based indexing in plt.get_fignums()

        # Assign correctly: reward diff, uncertainty, JSD
        fig_reward_diff = plt.figure(fig_nums[0])
        fig_uncertainty = plt.figure(fig_nums[1])
        fig_jsd = plt.figure(fig_nums[2])

        # Composite each into the correct axis
        for fig_src, ax in zip([fig_reward_diff,fig_jsd, fig_uncertainty], [ax0, ax1, ax2]):
            fig_src.canvas.draw()
            buf = fig_src.canvas.buffer_rgba()
            ax.imshow(buf)
            ax.axis('off')
    else:
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 1, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])      # Reward diff
        ax2 = fig.add_subplot(gs[1, 0])      # Total uncertainty

        # Grab the three most-recent model-free plots: reward diff, uncertainty, JSD
        fig_nums = plt.get_fignums()[-3:]  # Include the one before the model-free plots
        fig_nums = fig_nums[-2:]           # Keep only the last three
        fig_nums = [x - 1 for x in fig_nums] # Adjust for 0-based indexing in plt.get_fignums()

        # Assign correctly: reward diff, uncertainty, JSD
        fig_reward_diff = plt.figure(fig_nums[0])
        fig_uncertainty = plt.figure(fig_nums[1])

        # Composite each into the correct axis
        for fig_src, ax in zip([fig_reward_diff, fig_uncertainty], [ax1, ax2]):
            fig_src.canvas.draw()
            buf = fig_src.canvas.buffer_rgba()
            ax.imshow(buf)
            ax.axis('off')


    

    # Add title and subtitle
    fig.suptitle("Simulated latent states using one parameter set", fontsize=16)
    subtitle = ("Simulations were based on the maximum of the pdf of the choice×RT distribution."
                if sim_using_max_pdf else
                f"Simulations were sampled from the pdf of the choice×RT distribution {number_samples_to_sim} times.")
    fig.text(0.5, 0.935, subtitle, ha='center', fontsize=10)

    # Close old figures to clean up
    for fignum in fig_nums:
        plt.close(plt.figure(fignum))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show(block=False)




## PARAMETER SWEEP
# Specify the parameters to hold fixed during the parameter sweep. One of these parameters will
# be written over with the value of param_name.
if run_param_sweep:
    n_runs       = 1                               # specify number of simulations to run for each set of parameters. Can leave at 1 if we are using the max pdf method (simulates a choice/rt based on the max pdf) instead of sampling from the distribution of choices/RTs.
    game_len   =  [5,9] #[5,9] or [9]                              # specify the game length; use brackets
    metric_fn    = "hi" # lambda df: compute_stats_for_specific_horizon_and_choice(df, game_len=game_len, trial_idx=trial_idx)  # Use game_len to control whether plotting H1 (5) or H5 (9) games. Use trial_idx to control which which free choice to consider (5 = first free choice)
    # ==================================================================
    results = stats_simulate_parameter_sweep(social_media_sample, settings, param_name, param_vals, base_params, n_runs, metric_fn, sim_using_max_pdf, number_samples_to_sim)

    # ------------------------------------------------------------------
    # Plot: Probability of choosing the option with the higher generative mean for each horizon across the parameter sweep.
    # ------------------------------------------------------------------
    # Make subtitle
    free_choice_label = f"Free Choice {trial_idx - 4}"  # since 5 => 1, 6 => 2, etc.
    
    plt.figure()
    # If every value in the stderr_prob_choose_high_mean column is NaN, skip the error bars. This happens when n_runs=1 since the standard error of a single value is NaN.
    if 5 in game_len:
        choose_high_generative_mean_h1_standard_error = None if results["std_prob_choose_high_mean_horizon_5"].isna().all() else results["std_prob_choose_high_mean_horizon_5"]
        plt.errorbar(results[param_name], results["mean_prob_choose_high_mean_horizon_5"],
                yerr=choose_high_generative_mean_h1_standard_error, marker="o", linestyle="-", capsize=4, label="H1")
    if 9 in game_len:
        choose_high_generative_mean_h5_standard_error = None if results["std_prob_choose_high_mean_horizon_9"].isna().all() else results["std_prob_choose_high_mean_horizon_9"]
        plt.errorbar(results[param_name], results["mean_prob_choose_high_mean_horizon_9"],
                    yerr=choose_high_generative_mean_h5_standard_error, marker="o", linestyle="-", capsize=4, label="H5")
    plt.xlabel(param_name)
    plt.ylabel("P(choose higher generative mean)")
    plt.title(f"{param_name} sweep: P(choose higher generative mean)")
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
        choose_high_info_h1_standard_error = None if results["std_prob_choose_high_info_horizon_5"].isna().all() else results["std_prob_choose_high_info_horizon_5"]
        plt.errorbar(results[param_name], results["mean_prob_choose_high_info_horizon_5"],
                    yerr=choose_high_info_h1_standard_error, marker="o", linestyle="-", capsize=4, label="H1")
    if 9 in game_len:
        choose_high_info_h5_standard_error = None if results["std_prob_choose_high_info_horizon_9"].isna().all() else results["std_prob_choose_high_info_horizon_9"]
        plt.errorbar(results[param_name], results["mean_prob_choose_high_info_horizon_9"],
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
    # Plot: Average RT for each horizon across the parameter sweep. Note I could also separate by RTs for high and low generative means.
    # ------------------------------------------------------------------
    plt.figure()
    # Handle missing stderr separately for both lines
    if 5 in game_len:
        avg_rt_h1_standard_error = None if results["std_rt_horizon_5"].isna().all() else results["std_rt_horizon_5"]
        plt.errorbar(results[param_name], results["mean_rt_horizon_5"],
                yerr=avg_rt_h1_standard_error, label="H1", marker="o", linestyle="-", capsize=4)
    if 9 in game_len:
        avg_rt_h5_standard_error  = None if results["std_rt_horizon_9"].isna().all()  else results["std_rt_horizon_9"]
        plt.errorbar(results[param_name], results["mean_rt_horizon_9"],
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

