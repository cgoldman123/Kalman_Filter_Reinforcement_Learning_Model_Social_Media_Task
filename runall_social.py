import sys, os, re, subprocess

result_stem = sys.argv[1]
experiment = sys.argv[2]
model = "UCB"

ssub_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/social_media/VB_scripts/run_social.ssub'

subject_list_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/social_media/social_media_prolific_IDs.csv'
subjects = []
with open(subject_list_path) as infile:
    for line in infile:
        if 'ID' not in line:
            subjects.append(line.strip())


room_type = ["Like", "Dislike"]
for room in room_type:
    results = result_stem + room + "/"
    
    if not os.path.exists(results):
        os.makedirs(results)
        print(f"Created results directory {results}")

    if not os.path.exists(f"{results}/logs"):
        os.makedirs(f"{results}/logs")
        print(f"Created results-logs directory {results}/logs")

    for subject in subjects:
        
        stdout_name = f"{results}/logs/SM-{model}-{room}_room-{subject}-%J.stdout"
        stderr_name = f"{results}/logs/SM-{model}-{room}_room-{subject}-%J.stderr"
        jobname = f'SM-{model}-{room}_room-{subject}'
        os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {subject} {results} {model} {room} {experiment}")

        print(f"SUBMITTED JOB [{jobname}]")

# python3 /media/labs/rsmith/lab-members/cgoldman/Wellbeing/social_media/VB_scripts/runall_social.py /media/labs/rsmith/lab-members/cgoldman/Wellbeing/social_media/output/SM_fits_UCB_model_prolific_11-1-24/ "prolific"