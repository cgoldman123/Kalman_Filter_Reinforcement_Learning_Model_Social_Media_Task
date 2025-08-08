# Horizon Task: Behavioral Model Fitting Pipeline

## `Social_wrapper.m`

`Social_wrapper.m` is a flexible MATLAB wrapper function designed to process empirical or simulated behavioral data from social decision-making tasks. It supports data from multiple studies, performs model-based or model-free analyses, and can fit a variety of Bayesian and sequential sampling models to participant choice and reaction time data.

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

Open MATLAB and run:

```matlab
output_table = Social_wrapper();
```

OR

[@carter can you add a cluster usage eg]


* Specify whether to process **empirical** or **simulated** choices based on the `EMPIRICAL` flag. For **empirical** data, specify if 
* Specify study type: *wellbeing*, *exercise*, *adm*, *cobre_neut*, and *eit*. For *wellbeing*, also specify *local* or *prolific* in `study_info.experiment`
* [@Carter can you specify how to run this for either "ALL" participants or list of participants]
* Specify model type ` model` and which parameters to fit `MDP.field`
* Specify the result directory" `results_dir` 
* Load the appropriate dataset and parameters
* Fit or simulate the specified model
* Run optional plotting or model-free analyses
* Save trial-level output and return summary results in `output_table`

---

## Configuration

### Data Mode

Toggle between empirical or simulated analyses by modifying:

```matlab
EMPIRICAL = 1; % 1 = use empirical choices, 0 = simulate choices
```
For **empirical** data you can further specify if descriptive model-free analyses and/or model fitting is required by toggling [1: on; 0: off] `MDP.do_model_free` and `MDP.fit_model`, respectively.

If fitting model, futher options are available:
    1. ` MDP.do_simulated_model_free`: descriptive analyses of simulated behavior. RTs are not simlated for choice-only models. Associated fields are expected to be NaNs 
    2. `MDP.plot_fitted_behavior`: participant level plots, see section on plots below for details

For **simulated** data:
    1.  `MDP.param_to_sweep`;`MDP.param_values_to_sweep_over`:Specify the values of the parameter to sweep overspecify the name of the parameter name, and value range to sweep over or leave this empty to not sweep.
    2. If plotting simulated data, specify 
        - `gen_mean_difference`: % choose a generative mean difference
        - `horizon`: choose horizon of 1 or 5
        - `truncate_big_hor`: if truncate_big_hor is true, use the big bandit schedule but truncate so that all games are H1
    3. `MDP.num_samples_to_draw_from_pdf`: If 0, the model will simulate a choice/RT based on the maximum of the simulated pdf. If >0, it will sample from the distribution of choices/RTs this many times. Note this only matters for models that generate RTs.

### Study and Subject Info

Edit the `study_info` struct to specify:

* Study type: `'wellbeing'`, `'exercise'`, `'adm'`, etc.
* Subject ID
* Session/run (`T1`, `T2`, etc.)
* Room/context

For example:

```matlab
study_info.study = 'exercise';
study_info.id = 'AK465';
study_info.run = 'T1';
study_info.room = 'Like';
```

Cluster users can define these via environment variables (`RESULTS`, `ID`, `MODEL`, etc.).

### Model Specification

Supported models:

* `KF_SIGMA`, `KF_SIGMA_DDM`, `KF_SIGMA_RACING`
* `KF_SIGMA_logistic`, `KF_SIGMA_logistic_DDM`, `KF_SIGMA_logistic_RACING`
* `obs_means_logistic`, `obs_means_logistic_DDM`, `obs_means_logistic_RACING`

Model implementation scripts are located in `./SPM_models/`.

Set the model in:

```matlab
model = 'KF_SIGMA';
```

Each model has a pre-defined parameter structure and number of choices to fit (e.g., first choice only vs. all 5 choices). These are assigned automatically in the wrapper. Note: The `KF_SIGMA`, `KF_SIGMA_DDM`, and `KF_SIGMA_RACING` models can fit either the first free choice (`1`) or all five choices (`5`). This is specified via `MDP.num_choices_to_fit`.


---

### Model Description
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
[@Carter add file names]

---

## Dependencies

Make sure the following folders are in your MATLAB path:

* `./SPM_models/`: stores models
* `./data_processing/`: stores study specific raw data processing scripts
* `./racing_accumulator/`: specific function required for `racing_accumulator` models
* `./plotting/`: plotting scripts
* SPM12 (with `DEM` toolbox)

---

## Notes

* You do **not** need to edit below the "DO NOT EDIT" line unless modifying models themselves.
* Ensure `process_data_across_studies()` and model-fitting functions (e.g., `get_fits()`) are available in your path.
* Random seed is set for reproducibility (`rng(23)`).

---

