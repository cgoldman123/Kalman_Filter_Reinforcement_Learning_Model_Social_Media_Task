# Horizon Task: Behavioral Model Fitting Pipeline

## `Social_wrapper.m`

`Social_wrapper.m` is a flexible MATLAB wrapper function designed to process empirical or simulated behavioral data from the Horizon Task (https://psycnet.apa.org/record/2014-44776-001). It supports data from multiple studies, performs model-based or model-free analyses, and can fit a variety of logistic and sequential sampling models to participant choice and reaction time data.

## Features

* Supports both **empirical** and **simulated** analyses
* Fits several variants of **Kalman Filter (KF)** and **observed mean** models with logistic, DDM, or racing decision processes.
* Option to run **model-free** analyses on both empirical and simulated data. Toggle options described and available within `Social_wrapper.m`
* Modular processing pipeline currently supporting the following studies: *wellbeing*, *exercise*, *adm*, *cobre_neut*, and *eit*. This is specified in `study_info.study`. Note, for *wellbeing*, further specify if *local* or *prolific* data within `study_info.experiment`
* Automatically routes to appropriate data preprocessing scripts
* Generates trial-by-trial behavioral and model-based latent output
* Compatible with local and cluster-based environments.

---

## Usage

Open MATLAB and run the following script to sim/fit data for one subject:

```matlab
output_table = Social_wrapper();
```

* Specify whether to process **empirical** or **simulated** choices based on the `EMPIRICAL` flag. 
    * For **empirical** data, specify if you would like to fit the model to participants' actual data and/or conduct model-free analyses on this data.
        * If fitting the model to participant behavior, additionally specify if you would like to conduct model free analyses on data simulated using the posterior parameter estimates of the model. You may also plot the results of fitting the model and save trial-by-trial output, including model latents (e.g., prediction errors) and behavioral data (e.g., reaction times).
    * For **simulated** data, specify if you would like to plot model statistics, do model free analyses on simulated data (using the parameter values in Social_wrapper.m), and/or plot behavior for specific game types (i.e., MDP.do_plot_choice_given_gen_mean)
        * If plotting model statistics, indicate if you would like to conduct a parameter sweep by listing a parameter in MDP.param_to_sweep. Otherwise, leave it empty.
        * If plotting a specific game type, specify the horizon and generative mean difference of that game. Use truncate_big_hor to indicate if you would like to use the schedule of big horizon games but have the model treat them as small horizon games. 
* Use MDP.num_samples_to_draw_from_pdf = 0 to specify whether to simulate a choice/RT based on the maximum of the simulated pdf. If >0, it will sample from the distribution of choices RTs this many times. Note this only matters for models that generate RTs.

* Ensure the root directory is the root file path in your computer's file system.
* Specify study: *wellbeing*, *exercise*, *adm*, *cobre_neut*, and *eit*. For *wellbeing*, also specify *local* or *prolific* in `study_info.experiment`
* Within the study you are running, specify the subject you would like to fit and any additional fields to identify that subject's data. Additional fields include:
    * room: applies to horizon task studies with like (positive) and dislike (negative) rooms
    * run: applies to studies with multiple runs
    * cb: applies to studies with multiple counterbalanced schedules
    * experiment: applies to the wellbeing study, with an online and local experiment
    * condition: applies to the adm study with loaded (breathing resistance) and unloaded (without breathing resistance) conditions

* Specify model type ` model` and which parameters to fit `MDP.field`
* Specify the result directory" `results_dir` 

* For fitting, specify the prior parameter values in MDP.params. 
* For reaction time models, additionally specify the maximum reaction time to fit (max.rt), disregarding greater rts
* For certain models, indicate the number of free choices you would like to fit. Do not do this for models with "obs_mean" or "logistic" in the name, as they were not intended to be fit to multiple choices.



Supported models:

* `KF_SIGMA`, `KF_SIGMA_DDM`, `KF_SIGMA_RACING`
* `KF_SIGMA_logistic`, `KF_SIGMA_logistic_DDM`, `KF_SIGMA_logistic_RACING`
* `obs_means_logistic`, `obs_means_logistic_DDM`, `obs_means_logistic_RACING`

Model implementation scripts are located in `./SPM_models/`.

Each model has a pre-defined parameter structure and number of choices to fit (e.g., first choice only vs. all 5 choices). These are assigned automatically in the wrapper. Note: The `KF_SIGMA`, `KF_SIGMA_DDM`, and `KF_SIGMA_RACING` models can fit either the first free choice (`1`) or all five choices (`5`). This is specified via `MDP.num_choices_to_fit`.

---

### Model Free

* Note that in model free analysis, `has_equal_info_games`: whether indicate whether a given dataset has [2 2] games 
suffix _13 or _22 correspond to games types [1 3] games or [2 2] games. applicable to all the statistics

### Model Descriptions


1. **`KF_SIGMA_DDM`: Kalman filter + drift-diffusion model for jointly modeling choice and RTs**
    - Fit/sim first free choice (`1`) or all choices (`5`)

    - Required Fields in `params`:
        * `initial_mu`, `initial_sigma`: Prior beliefs about bandit means and uncertainty
        * `reward_sensitivity`: Scaling of rewards
        * `sigma_r`, `sigma_d`: Observation and diffusion noise in Kalman updates
        * `side_bias`: Static bias for one option
        * `baseline_noise`: Inverse temperature
        * `cong_base_info_bonus`, `incong_base_info_bonus`: Baseline directed exploration bonuses when information is congruent/incongruent 
        * `cong_directed_exp`, `incong_directed_exp`: Dynamic exploration bonuses modulated by trials remaining
        * `random_exp`: Horizon-scaled random noise influencing value integration
        * `decision_thresh_baseline`: Decision threshold for the drift-diffusion process 
        * `rdiff_bias_mod`: Bias based on reward difference

    - Returns a struct `model_output` with the following fields:
        * **Behavioral accuracy and probabilities**:
        `actions`, `rewards`, `action_probs`, `model_acc`
        * **Learning signals**:
        * `exp_vals`: Expected value of the chosen option
        * `pred_errors`: Reward prediction error
        * `pred_errors_alpha`: Scaled prediction error (learning signal)
        * `alpha`: Learning rate
        * **Uncertainty tracking**:
        * `sigma1`, `sigma2`: Uncertainty estimates for each bandit
        * `relative_uncertainty_of_choice`: Between-bandit uncertainty at time of choice
        * `total_uncertainty`: Combined uncertainty
        * `change_in_uncertainty_after_choice`: Trial-wise change in chosen bandit's uncertainty
        * **Derived value signals**
        * `estimated_mean_diff`: Value difference between options
        * **Reaction times**
        * `rts`: Observed or simulated response times
        * `rt_pdf`: Likelihood of observed RT under the drift-diffusion model
        * `num_invalid_rts`: Number of trials with invalid RTs (e.g., RT ≤ 0 or RT ≥ max RT)

2. **`KF_SIGMA`: Kalman filter-based choice model**
    - Fit/sim first free choice (`1`) or all choices (`5`)

    - Required Fields in `params`:
        * `initial_mu`, `initial_sigma`: Prior beliefs about bandit means and uncertainty
        * `reward_sensitivity`: Scaling of rewards
        * `sigma_r`, `sigma_d`: Observation and diffusion noise in Kalman updates
        * `side_bias`: Static bias for one option
        * `baseline_noise`: inverse temperature
        * `cong_base_info_bonus`, `incong_base_info_bonus`: Baseline directed exploration bonuses when information is congruent/incongruent 
        * `cong_directed_exp`, `incong_directed_exp`: Dynamic exploration bonuses modulated by trials remaining
        * `random_exp`: Horizon-scaled random noise influencing value integration

    - Returns a struct `model_output` with the following fields:
        * **Behavioral accuracy and probabilities**: 
            `actions`, `rewards`, `action_probs`, `model_acc`
        * **Learning signals**:
            * `exp_vals`: Expected value of the chosen option
            * `pred_errors`: Reward prediction error
            * `pred_errors_alpha`: Scaled prediction error (learning signal)
            * `alpha`: Learning rate
        * **Uncertainty tracking**:
            * `sigma1`, `sigma2`: Uncertainty estimates for each bandit
            * `relative_uncertainty_of_choice`: Between-bandit uncertainty at time of choice
            * `total_uncertainty`: Combined uncertainty
            * `change_in_uncertainty_after_choice`: Trial-wise change in chosen bandit's uncertainty
        * **Derived value signals**
            * `estimated_mean_diff`: Value difference between options
        * **Reaction times**
            * `rts`: (NaNs for this model, which does not predict RTs)

3. **`KF_SIGMA_RACING`: Kalman filter + Racing DDM**
    - Fit/sim first free choice (`1`) or all choices (`5`)

    - Required Fields in `params`:
        * `initial_mu`, `initial_sigma`: Prior beliefs about bandit means and uncertainty
        * `reward_sensitivity`: Scaling of rewards
        * `sigma_r`, `sigma_d`: Observation and diffusion noise in Kalman updates
        * `side_bias`: Side preference
        * `baseline_noise`: inverse temperature
        * `cong_base_info_bonus`, `incong_base_info_bonus`: Baseline directed exploration bonuses when information is congruent/incongruent 
        * `cong_directed_exp`, `incong_directed_exp`: Dynamic exploration bonuses modulated by trials remaining
        * `random_exp`: Horizon-scaled random noise influencing value integration
        * `decision_thresh_baseline`: Decision threshold for the drift-diffusion process
        * `starting_bias_baseline`: Starting point bias (centered at 0.5 if neutral)
        * `drift_baseline`: Baseline drift rate
        * `drift_reward_diff_mod`: Influence of reward difference on drift rate 
        * `starting_bias_reward_diff_mod`: Influence of reward difference on starting bias
        * `wd`: Weight on reward difference in racing model
        * `ws`: Weight on total expected value in racing model
        * `V0`: Baseline drift rate

    - Returns a struct `model_output` with the following fields:
        * **Behavioral accuracy and probabilities**:
        `actions`, `rewards`, `action_probs`, `model_acc`
        * **Learning signals**:
        * `exp_vals`: Expected value of the chosen option
        * `pred_errors`: Reward prediction error
        * `pred_errors_alpha`: Scaled prediction error (learning signal)
        * `alpha`: Learning rate
        * **Uncertainty tracking**:
        * `sigma1`, `sigma2`: Uncertainty estimates for each bandit
        * `relative_uncertainty_of_choice`: Between-bandit uncertainty at time of choice
        * `total_uncertainty`: Combined uncertainty
        * `change_in_uncertainty_after_choice`: Trial-wise change in chosen bandit's uncertainty
        * **Derived value signals**
        * `estimated_mean_diff`: Value difference between options
        * **Reaction times**
        * `rts`: Observed or simulated response times
        * `rt_pdf`: Likelihood of observed RT under the drift-diffusion model
        * `num_invalid_rts`: Number of trials with invalid RTs (e.g., RT ≤ 0 or RT ≥ max RT)


4. **`KF_SIGMA_logistic`: Kalman filter + logistic choice policy**
    - This model is for the first free choice only

    - Required Fields in `params`:
        * `initial_mu`, `initial_sigma`: Prior beliefs about bandit means and uncertainty
        * `reward_sensitivity`: Scaling of rewards
        * `sigma_r`, `sigma_d`: Observation and diffusion noise in Kalman updates
        * `info_bonus_small_hor`, `info_bonus_big_hor`: Information bonus
        * `dec_noise_small_hor`, `dec_noise_big_hor`: Inverse temperature
        * `side_bias_small_hor`, `side_bias_big_hor`: Side biases

    - Returns a struct `model_output` with the following fields:
        * **Behavioral accuracy and probabilities**:
        `actions`, `rewards`, `action_probs`, `model_acc`
        * **Learning signals**:
        * `exp_vals`: Expected value of the chosen option
        * `pred_errors`: Reward prediction error
        * `pred_errors_alpha`: Scaled prediction error (learning signal)
        * `alpha`: Learning rate
        * **Uncertainty tracking**:
        * `sigma1`, `sigma2`: Uncertainty estimates for each bandit
        * `relative_uncertainty_of_choice`: Between-bandit uncertainty at time of choice
        * `total_uncertainty`: Combined uncertainty
        * `change_in_uncertainty_after_choice`: Trial-wise change in chosen bandit's uncertainty
        * **Derived value signals**
        * `estimated_mean_diff`: Value difference between options
        * **Reaction times**
        * `rts`: (NaN-filled; this model does not predict RTs)


5. **`KF_SIGMA_logistic_DDM`: Kalman filter + logistic choice policy + DDM**
    - This model is for the first free choice only

    - Required Fields in `params`:
        * `initial_mu`, `initial_sigma`: Prior beliefs about bandit means and uncertainty
        * `reward_sensitivity`: Scaling of rewards
        * `sigma_r`, `sigma_d`: Observation and diffusion noise in Kalman updates
        * `info_bonus_small_hor`, `info_bonus_big_hor`: Information bonus
        * `dec_noise_small_hor`, `dec_noise_big_hor`: Inverse temperature
        * `side_bias_small_hor`, `side_bias_big_hor`: Side biases
        * `decision_thresh_baseline`: Decision threshold for the drift-diffusion process
        * `starting_bias_baseline`: Starting point bias (centered at 0.5 if neutral)
        * `drift_baseline`: Baseline drift rate
        * `drift_reward_diff_mod`: Influence of reward difference on drift rate
        * `starting_bias_reward_diff_mod`: Influence of reward difference on starting bias

    - Returns a struct `model_output` with the following fields:
        * **Behavioral accuracy and probabilities**:
        `actions`, `rewards`, `action_probs`, `model_acc`
        * **Learning signals**:
        * `exp_vals`: Expected value of the chosen option
        * `pred_errors`: Reward prediction error
        * `pred_errors_alpha`: Scaled prediction error (learning signal)
        * `alpha`: Learning rate
        * **Uncertainty tracking**:
        * `sigma1`, `sigma2`: Uncertainty estimates for each bandit
        * `relative_uncertainty_of_choice`: Between-bandit uncertainty at time of choice
        * `total_uncertainty`: Combined uncertainty
        * `change_in_uncertainty_after_choice`: Trial-wise change in chosen bandit's uncertainty
        * **Derived value signals**
        * `estimated_mean_diff`: Value difference between options
        * **Reaction times**
        * `rts`: Observed or simulated response times
        * `rt_pdf`: Likelihood of observed RT under the drift-diffusion model
        * `num_invalid_rts`: Number of trials with invalid RTs (e.g., RT ≤ 0 or RT ≥ max RT)


6. **`KF_SIGMA_logistic_RACING`: Kalman filter + logistic choice policy + Racing DDM**
    - This model is for the first free choice only

    - Required Fields in `params`:
        * `initial_mu`, `initial_sigma`: Prior beliefs about bandit means and uncertainty
        * `reward_sensitivity`: Scaling of rewards
        * `sigma_r`, `sigma_d`: Observation and diffusion noise in Kalman updates
        * `info_bonus_small_hor`, `info_bonus_big_hor`: Information bonus
        * `dec_noise_small_hor`, `dec_noise_big_hor`: Inverse temperature
        * `side_bias_small_hor`, `side_bias_big_hor`: Side biases
        * `decision_thresh_baseline`: Decision threshold for the drift-diffusion process
        * `starting_bias_baseline`: Starting point bias (centered at 0.5 if neutral)
        * `drift_baseline`: Baseline drift rate
        * `drift_reward_diff_mod`: Influence of reward difference on drift rate
        * `starting_bias_reward_diff_mod`: Influence of reward difference on starting bias
        * `wd`: Weight on reward difference in racing model
        * `ws`: Weight on total expected value in racing model
        * `V0`: Baseline drift rate

    - Returns a struct `model_output` with the following fields:
        * **Behavioral accuracy and probabilities**:
        `actions`, `rewards`, `action_probs`, `model_acc`
        * **Learning signals**:
        * `exp_vals`: Expected value of the chosen option
        * `pred_errors`: Reward prediction error
        * `pred_errors_alpha`: Scaled prediction error (learning signal)
        * `alpha`: Learning rate
        * **Uncertainty tracking**:
        * `sigma1`, `sigma2`: Uncertainty estimates for each bandit
        * `relative_uncertainty_of_choice`: Between-bandit uncertainty at time of choice
        * `total_uncertainty`: Combined uncertainty
        * `change_in_uncertainty_after_choice`: Trial-wise change in chosen bandit's uncertainty
        * **Derived value signals**
        * `estimated_mean_diff`: Value difference between options
        * **Reaction times**
        * `rts`: Observed or simulated response times
        * `rt_pdf`: Likelihood of observed RT under the drift-diffusion model
        * `num_invalid_rts`: Number of trials with invalid RTs (e.g., RT ≤ 0 or RT ≥ max RT)


7. **`obs_means_logistic`: Logistic choice model based on observed mean rewards**
    - This model is for the first free choice only

    - Required Fields in `params`:
        * `initial_mu`: Prior beliefs about bandit means
        * `reward_sensitivity`: Scaling of rewards
        * `side_bias_small_hor`, `side_bias_big_hor`: Side biases
        * `info_bonus_small_hor`, `info_bonus_big_hor`: Information bonus
        * `dec_noise_small_hor`, `dec_noise_big_hor`: Inverse temperature

    - Returns a struct `model_output` with the following fields:
        * **Behavioral accuracy and probabilities**:
        `actions`, `rewards`, `action_probs`, `model_acc`
        * **Learning signals**:
        * `exp_vals`: Expected value of the chosen option
        * `pred_errors`: Reward prediction error
        * `pred_errors_alpha`: Scaled prediction error (learning signal)
        * `alpha`: Learning rate
        * **Uncertainty tracking**:
        * `sigma1`, `sigma2`: Uncertainty estimates for each bandit
        * `relative_uncertainty_of_choice`: Between-bandit uncertainty at time of choice
        * `total_uncertainty`: Combined uncertainty
        * `change_in_uncertainty_after_choice`: Trial-wise change in chosen bandit's uncertainty
        * **Derived value signals**
        * `estimated_mean_diff`: Value difference between options
        * **Reaction times**
        * `rts`: (NaNs for this model, which does not predict RTs)


8. **`obs_means_logistic_DDM`: Logistic choice model based on observed mean rewards + DDM**
    - This model is for the first free choice only.

    - Required Fields in `params`:
        * `initial_mu`: Prior beliefs about bandit means
        * `reward_sensitivity`: Scaling of rewards
        * `side_bias_small_hor`, `side_bias_big_hor`: Side biases
        * `info_bonus_small_hor`, `info_bonus_big_hor`: Information bonus
        * `dec_noise_small_hor`, `dec_noise_big_hor`: Inverse temperature
        * `decision_thresh_baseline`: Decision threshold for the drift-diffusion process
        * `starting_bias_baseline`: Starting point bias (centered at 0.5 if neutral)
        * `drift_baseline`: Baseline drift rate
        * `drift_reward_diff_mod`: Influence of reward difference on drift rate
        * `starting_bias_reward_diff_mod`: Influence of reward difference on starting bias

    - Returns a struct `model_output` with the following fields:
        * **Behavioral accuracy and probabilities**:
        `actions`, `rewards`, `action_probs`, `model_acc`
        * **Learning signals**:
        * `exp_vals`: Expected value of the chosen option
        * `pred_errors`: Reward prediction error
        * `pred_errors_alpha`: Scaled prediction error (learning signal)
        * `alpha`: Learning rate
        * **Uncertainty tracking**:
        * `sigma1`, `sigma2`: Uncertainty estimates for each bandit
        * `relative_uncertainty_of_choice`: Between-bandit uncertainty at time of choice
        * `total_uncertainty`: Combined uncertainty
        * `change_in_uncertainty_after_choice`: Trial-wise change in chosen bandit's uncertainty
        * **Derived value signals**
        * `estimated_mean_diff`: Value difference between options
        * **Reaction times**
        * `rts`: Observed or simulated response times
        * `rt_pdf`: Likelihood of observed RT under the drift-diffusion model
        * `num_invalid_rts`: Number of trials with invalid RTs (e.g., RT ≤ 0 or RT ≥ max RT)
        * `decision_thresh`: Decision threshold used   


9. **`obs_means_logistic_RACING`: Logistic choice model based on observed mean rewards + Racing DDM**
    - This model is for the first free choice only.

    - Required Fields in `params`:
        * `initial_mu`: Prior beliefs about bandit means
        * `reward_sensitivity`: Scaling of rewards
        * `side_bias_small_hor`, `side_bias_big_hor`: Side biases
        * `info_bonus_small_hor`, `info_bonus_big_hor`: Information bonus
        * `dec_noise_small_hor`, `dec_noise_big_hor`: Inverse temperature
        * `decision_thresh_baseline`: Decision threshold for the drift-diffusion process
        * `starting_bias_baseline`: Starting point bias (centered at 0.5 if neutral)
        * `drift_baseline`: Baseline drift rate
        * `drift_reward_diff_mod`: Influence of reward difference on drift rate
        * `starting_bias_reward_diff_mod`: Influence of reward difference on starting bias
        * `wd`: Weight on reward difference in racing model
        * `ws`: Weight on total expected value in racing model
        * `V0`: Baseline drift rate

    - Returns a struct `model_output` with the following fields:
        * **Behavioral accuracy and probabilities**:
        `actions`, `rewards`, `action_probs`, `model_acc`
        * **Learning signals**:
        * `exp_vals`: Expected value of the chosen option
        * `pred_errors`: Reward prediction error
        * `pred_errors_alpha`: Scaled prediction error (learning signal)
        * `alpha`: Learning rate
        * **Uncertainty tracking**:
        * `sigma1`, `sigma2`: Uncertainty estimates for each bandit
        * `relative_uncertainty_of_choice`: Between-bandit uncertainty at time of choice
        * `total_uncertainty`: Combined uncertainty
        * `change_in_uncertainty_after_choice`: Trial-wise change in chosen bandit's uncertainty
        * **Derived value signals**
        * `estimated_mean_diff`: Value difference between options
        * **Reaction times**
        * `rts`: Observed or simulated response times
        * `rt_pdf`: Likelihood of observed RT under the drift-diffusion model
        * `num_invalid_rts`: Number of trials with invalid RTs (e.g., RT ≤ 0 or RT ≥ max RT)
        * `decision_thresh`: Decision threshold used   


---

## Output

Depending on the mode, the wrapper generates:

* Trial-by-trial behavioral data
* Fitted model latent variables (e.g., prediction errors, Q-values)
* Plots of model behavior or parameter sweeps
* Final summary `output_table` containing results of interest

Files are saved to the `results_dir` specified near the top of the script.

### `output_table` 



### Plots

(a) *top*: Mean rt by reward difference; *bottom*: Mean rt by reward difference for high - low info option
(b) *top*: Total Uncertainty by Choice Number; *bottom*: Estimated Reward Difference by Choice Number
(c) *top*: Probability of Correct Choice; *middle*: Reaction Time; *bottom*: Probability of Choosing High-Info Option
(d) *top*: P(Choose Right) by Reward Difference; *bottom*: P(Choose High-Info) by Reward Difference

---

## Dependencies

Make sure that you have SPM12 installed and added to your MATLAB path. Additionally add the `DEM` toolbox within SPM.

---


## Legacy Scripts

The `/legacy` folder contains alternative and historical analysis pipelines, including both MATLAB and Python implementations for fitting and simulating behavioral models. In the legacy folder, Social_wrapper.m is the starting point for fitting models using these methods. 

### `/legacy/PyDDM_scripts/`

This folder contains Python scripts for modeling the Horizon Task using the [`pyddm`](https://pyddm.readthedocs.io/en/latest/) package:

* **`fit_pyddm_social_media.py`** – Loads behavioral data from the Social Media variant of the Horizon Task, reformats it into a `pyddm.Sample`, and fits a **Kalman Filter + Drift Diffusion Model (KF-DDM)** to choice and reaction time data.  
  * Supports different fitting settings (e.g., fit all RTs vs. first 3 RTs; fit using JSD or z-scored RTs).
  * Saves fitted parameter estimates, model outputs (choice probabilities, RT PDFs, prediction errors, uncertainty measures), and simulation recoverability results to `.mat` files for further analysis.
  * Produces trial-by-trial simulated behavior based on the fitted parameters.

* **`simulate_pyddm_social_media.py`** – Simulates behavior from a KF-DDM model without fitting to new data, using predefined or swept parameter values.  
  * Generates and plots model-free statistics such as mean RT by reward difference, choice number, and horizon.
  * Can sweep over a specified model parameter to examine its effect on choice behavior, information seeking, and RTs.
  * Includes options to:
    * Simulate using the **maximum of the predicted choice×RT distribution** or sampling multiple times from the distribution.
    * Plot latent states (e.g., total uncertainty, reward difference, Jensen–Shannon divergence) by horizon and choice number.

### VBA Toolbox Models

In addition to the PyDDM pipelines, the `/legacy` folder includes MATLAB scripts for fitting models using the [VBA Toolbox](https://mbb-team.github.io/VBA-toolbox/).  
The VBA Toolbox implements **Variational Bayes Analysis** for fitting hierarchical and single-subject models, providing a flexible alternative to PyDDM for the Horizon Task data.

---


