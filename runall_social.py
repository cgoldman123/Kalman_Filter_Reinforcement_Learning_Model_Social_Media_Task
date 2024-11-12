import sys, os, re, subprocess
from datetime import datetime

result_stem = sys.argv[1]
experiment = sys.argv[2]
model_class = "UCB"

current_datetime = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
result_stem = f"{result_stem}_{current_datetime}/"

ssub_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/social_media/VB_scripts/run_social.ssub'

subject_list_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/social_media/social_media_prolific_IDs.csv'
subjects = []
with open(subject_list_path) as infile:
    for line in infile:
        if 'ID' not in line:
            subjects.append(line.strip())



models = [
    {'field': 'baseline_noise,side_bias,sigma_r'},
    {'field': 'baseline_noise,side_bias,sigma_r,info_bonus'},
    {'field': 'baseline_noise,side_bias,sigma_r,baseline_info_bonus'},
    {'field': 'baseline_noise,side_bias,sigma_r,random_exp'},
    {'field': 'baseline_noise,side_bias,sigma_r,reward_sensitivity'},
    {'field': 'baseline_noise,side_bias,sigma_r,info_bonus,baseline_info_bonus'},
    {'field': 'baseline_noise,side_bias,sigma_r,info_bonus,random_exp'},
    {'field': 'baseline_noise,side_bias,sigma_r,info_bonus,reward_sensitivity'},
    {'field': 'baseline_noise,side_bias,sigma_r,baseline_info_bonus,random_exp'},
    {'field': 'baseline_noise,side_bias,sigma_r,baseline_info_bonus,reward_sensitivity'},
    {'field': 'baseline_noise,side_bias,sigma_r,random_exp,reward_sensitivity'},
    {'field': 'baseline_noise,side_bias,sigma_r,info_bonus,baseline_info_bonus,random_exp'},
    {'field': 'baseline_noise,side_bias,sigma_r,info_bonus,baseline_info_bonus,reward_sensitivity'},
    {'field': 'baseline_noise,side_bias,sigma_r,info_bonus,random_exp,reward_sensitivity'},
    {'field': 'baseline_noise,side_bias,sigma_r,baseline_info_bonus,random_exp,reward_sensitivity'},
    {'field': 'baseline_noise,side_bias,sigma_r,info_bonus,baseline_info_bonus,random_exp,reward_sensitivity'},

    {'field': 'baseline_noise,side_bias,sigma_r,DE_RE_horizon'},
    {'field': 'baseline_noise,side_bias,sigma_r,DE_RE_horizon,baseline_info_bonus'},
    {'field': 'baseline_noise,side_bias,sigma_r,DE_RE_horizon,reward_sensitivity'},
    {'field': 'baseline_noise,side_bias,sigma_r,DE_RE_horizon,baseline_info_bonus,reward_sensitivity'},
]


room_type = ["Like", "Dislike"]
for room in room_type:
    results = result_stem + room + "/"

    for index, model in enumerate(models, start=1):
        # test four additional models
        if index < 17:
            continue
    
        combined_results_dir = os.path.join(results, f"model{index}/")
        field = model['field']
        
        if not os.path.exists(combined_results_dir):
            os.makedirs(combined_results_dir)
            print(f"Created results directory {combined_results_dir}")

        if not os.path.exists(f"{combined_results_dir}/logs"):
            os.makedirs(f"{combined_results_dir}/logs")
            print(f"Created results-logs directory {combined_results_dir}/logs")
        # i = 0
        for subject in subjects:
            
            stdout_name = f"{combined_results_dir}/logs/SM-{model_class}-model_{index}-{room}_room-{subject}-%J.stdout"
            stderr_name = f"{combined_results_dir}/logs/SM-{model_class}-model_{index}-{room}_room-{subject}-%J.stderr"
            jobname = f'SM-{model_class}-model_{index}-{room}_room-{subject}'
            os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {subject} {combined_results_dir} {room} {experiment} {field}")

            print(f"SUBMITTED JOB [{jobname}]")
            # i = i+1
            # if i ==4:
            #     break


# python3 /media/labs/rsmith/lab-members/cgoldman/Wellbeing/social_media/VB_scripts/runall_social.py /media/labs/rsmith/lab-members/cgoldman/Wellbeing/social_media/output/SM_fits_UCB_model_DE_RE_combined "prolific"