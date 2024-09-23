import sys, os, re, subprocess

subject_list_path = sys.argv[1]
ses = sys.argv[2]
input_dir = sys.argv[3]
room_type = sys.argv[4]
results = sys.argv[5]
counterbalance = sys.argv[6]


if not os.path.exists(results):
    os.makedirs(results)
    print(f"Created results directory {results}")

if not os.path.exists(f"{results}/logs"):
    os.makedirs(f"{results}/logs")
    print(f"Created results-logs directory {results}/logs")

subjects = []
with open(subject_list_path) as infile:
    for line in infile:
        if 'ID' not in line:
            subjects.append(line.strip())

ssub_path = '/media/labs/rsmith/wellbeing/tasks/SocialMedia/scripts/run_social_old.ssub'

for subject in subjects:
    stdout_name = f"{results}/logs/social-test-%J-{subject}.stdout"
    stderr_name = f"{results}/logs/social-test-%J-{subject}.stderr"
    jobname = f'social-oldmodel-{subject}-fit'
    os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {subject} {ses} {input_dir} {room_type} {results} {counterbalance}")

    print(f"SUBMITTED JOB [{jobname}]")

# python3 runall_social_old.py {path location of ids} {1 or 2 depending on test or retest} {DataSink directory} {Like or Dislike} {desired output directory} {1 or 2}

