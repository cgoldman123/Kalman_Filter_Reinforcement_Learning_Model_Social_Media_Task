import pyddm
import pandas as pd
import numpy as np


########### Load in Social Media data and format as Sample object ###########
with open("L:/rsmith/lab-members/cgoldman/Wellbeing/social_media/output/test/60f8cc430a3d06de34a78773_beh_Like_04_01_25_T10-24-15.csv", "r") as f:
    df = pd.read_csv(f)

# Extract trial numbers
trial_nums = list(range(1, 10)) # A list from 1 to 9

# Create a tidy DataFrame by stacking trial-wise columns
social_media_df = pd.DataFrame({
    'game_number': df['game'].repeat(9).values,
    'gameLength': df['gameLength'].repeat(9).values,
    'trial': trial_nums * len(df),
    'r': df[[f'r{i}' for i in trial_nums]].values.flatten(),
    'c': df[[f'c{i}' for i in trial_nums]].values.flatten() - 1,
    'rt': df[[f'rt{i}' for i in trial_nums]].values.flatten(),
})
social_media_df = social_media_df.dropna(subset=["c"])
# Replace NA values in rt col with -1 so it can be passed into sample object
social_media_df['rt'] = social_media_df['rt'].fillna(-1)


social_media_sample = pyddm.Sample.from_pandas_dataframe(social_media_df, rt_column_name="rt", choice_column_name="c", choice_names=("left", "right"))

# Example functions to get summary statistics. Not that these functions use all trials, including forced choices
social_media_sample.cdf("left",dt=.01,T_dur=2)
social_media_sample.condition_names()
social_media_sample.condition_values("game_number")
social_media_sample.items(choice="left")
social_media_sample.pdf("left", dt=.01, T_dur=2)
social_media_sample.prob("left")




class KF_DDM_Loss(pyddm.LossFunction):
    name = "KF_DDM_Loss"
    def setup(self, **kwargs):
         # This setup method organizes the data into a list of dataframes for each game number
         # and checks that the trials are sequentially numbered.
         self.game_number = self.sample.condition_values('game_number')
         for s in self.game_number:
             trials = self.sample.subset(game_number=s).condition_values('trial')
             assert set(trials) == set(range(min(trials), min(trials)+len(trials))), "Trials must be sequentially numbered"
         self.df =  self.sample.to_pandas_dataframe()
         self.sessdfs = [self.df.query(f'game_number == {s}').sort_values('trial') for s in self.game_number]
    def loss(self, model):
        likelihood = 0
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


        for game_num in range(0, len(self.game_number)):
            sessdf = self.sessdfs[game_num]
            # Initialize empty vectors for each game
            mu1 = np.full(10, np.nan)
            mu1[0] = 50 # Irrelevant when initial_sigma is 10000 since it makes the learning rate on the first trial 1
            mu2 = np.full(10, np.nan)
            mu2[0] = 50 # Irrelevant when initial_sigma is 10000 since it makes the learning rate on the first trial 1
            
            alpha1 = np.full(10, np.nan)
            alpha2 = np.full(10, np.nan)
        
            for trial_num in range(0,len(sessdf)):
                trial = sessdf.iloc[trial_num]
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

                    chose_left = trial['choice'] == 0
                    choice = "left" if chose_left else "right"

                    # Only consider reaction times less than the max rt in the log likelihood
                    if trial['RT'] < max_rt:
                        # solve a ddm (i.e., get the probability density function) for current DDM parameters
                        sol = model.solve_analytical(conditions={"reward_diff": reward_diff + side_bias})
                        # Evaluate the pdf of the reaction time for the chosen option. Note that left will be the bottom boundary and right upper
                        p = sol.evaluate(trial['RT'], not chose_left)
                        assert p >= 0, "Probability density of a reaction time must be non-negative"
                        # Add to the likelihood
                        likelihood += np.log(p)
                        # Store the probability of the choice under the model
                        rt_pdf[game_num, trial_num] = p
                        # Store the action probability of the choice under the model
                        action_probs[game_num, trial_num] = sol.prob(choice)
                        model_acc[game_num, trial_num] = sol.prob(choice) > 1
                    else:
                        num_invalid_rts += 1


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


        # Store model statistics
        model.action_probs = action_probs
        model.rt_pdf = rt_pdf
        model.model_acc = model_acc
        model.num_invalid_rts = num_invalid_rts
        model.exp_vals = exp_vals
        model.pred_errors = pred_errors
        model.pred_errors_alpha = pred_errors_alpha
        model.alpha = alpha
        model.sigma1 = sigma1
        model.sigma2 = sigma2
        model.relative_uncertainty_of_choice = relative_uncertainty_of_choice
        model.total_uncertainty = total_uncertainty
        model.change_in_uncertainty_after_choice = change_in_uncertainty_after_choice

        return -likelihood



# This is kind of hacky, but we pass in the learning parameters as arguments to drift even though they aren't used for it. This is just so the loss function has access to them
model_to_fit = pyddm.gddm(drift=lambda drift_reward_diff_mod,reward_diff,sigma_d,sigma_r,baseline_noise,side_bias,directed_exp,baseline_info_bonus,random_exp : drift_reward_diff_mod * reward_diff,
                          noise=1.0, bound="B", nondecision=0, starting_position="x0", T_dur=4.17,
                          conditions=["game_number", "gameLength", "trial", "r", "reward_diff"],
                          parameters={"drift_reward_diff_mod": (-2,2), "B": (0.3, 2), "x0": (-.8, .8), "sigma_d": (0,20), "sigma_r": (0,20), "baseline_noise": (1,10), "side_bias": (-2,2), "directed_exp": (-5,5), "baseline_info_bonus": (-5,5), "random_exp": (-5,5)}, choice_names=("left","right"))



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

pyddm.plot.model_gui(m, conditions={"deltaq": [-1, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, 1]})
# END MODEL


print("hi")

## Todo - be able to pass in fixed values like initial_sigma, starting value, reward sensitivity, max_rt, etc