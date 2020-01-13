#  used to remove extra distillation experiments
import os
import subprocess
import sys

exp_dir = sys.argv[1]
exps = os.listdir(exp_dir)
exp_paths = [os.path.join(exp_dir, exp) for exp in exps]

num_empty_exp = 0
for exp_path in exp_paths:
    ckpt_path = os.path.join(exp_path, 'checkpoints')
    if not os.path.exists(ckpt_path):  # no checkpoint folder
        print('no ckpt dir')
        print(os.listdir(exp_path))
        subprocess.call("rm -rf {}".format(exp_path), shell=True)
        num_empty_exp += 1
        continue

    ckpts = [f for f in os.listdir(ckpt_path) if 'ckpt' in f]
    if len(ckpts) < 4:  # small checkpoint folder
        subprocess.call("rm -rf {}".format(exp_path), shell=True)
        num_empty_exp += 1
    else:  # big checkpoint folder
        print('real exp')
        print(exp_path)
        print(ckpts)

print(f'killed {num_empty_exp} experiments')
