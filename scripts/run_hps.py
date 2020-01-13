import sys
LOG_DIR = sys.argv[1]  # must be run before import transfer because transfer will pop the argv
from scripts.train_transfer import ex
from tlkit.utils import flatten
import numpy as np
import os
import subprocess
from evkit.saving.observers import FileStorageObserverWithExUuid

@ex.command
def run_hps(cfg, uuid):
    print(cfg)

    # Get argv
    argv_plus_hps = sys.argv
    script_name = argv_plus_hps[0]
    script_name = script_name.replace('.py','').replace('/','.')
    script_name = script_name[1:] if script_name.startswith('.') else script_name

    # Sample and load HPS into argv
    for hp, hp_range in flatten(cfg['hps_kwargs']['hp']).items():
        hp_val = np.power(10, np.random.uniform(*hp_range))
        argv_plus_hps.append(f'cfg.{hp}={hp_val}')

    # Update argv script name and uuid
    argv_plus_hps = [a.replace('run_hps', cfg['hps_kwargs']['script_name']) for a in argv_plus_hps]
    argv_plus_hps.append(f'uuid={uuid}_hps_run')

    # Run real experiment
    print(f'python -m {script_name} {LOG_DIR} {" ".join(argv_plus_hps[1:])}')
    ex.run_commandline(argv=argv_plus_hps)


@ex.named_config
def cfg_hps():
    uuid='hps'
    cfg = {}
    cfg['hps_kwargs'] = {
        'hp': {
            # pass in hp like you would for regular run. but instead of a number, pass in a log exp range
            # (if not log range, we will need to update, maybe with explicit dictionaries)
            'learner': {
                'lr': (-5, -3),
                'optimizer_kwargs' : {
                    'weight_decay': (-6,-4)
                },
            },
        },
        'script_name': 'train',
        'add_time_to_logdir': True,  # TODO Should I edit logdir for uniqueness or do it manually? time, make it a parameter to do it or not
    }

if __name__ == '__main__':
    assert LOG_DIR, 'log dir cannot be empty'
    os.makedirs(LOG_DIR, exist_ok=True)
    subprocess.call("rm -rf {}/*".format(LOG_DIR), shell=True)
    ex.observers.append(FileStorageObserverWithExUuid.create(LOG_DIR))
    ex.run_commandline()
else:
    print(__name__)


