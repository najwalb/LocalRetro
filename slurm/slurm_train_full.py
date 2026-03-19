"""SLURM job: preprocess + train LocalRetro on USPTO_FULL (~810K reactions)."""
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from slurm_utils import get_platform_info, create_and_submit_batch_job

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]
DATASET = 'USPTO_FULL'
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Platform & SLURM config ---
slurm_args = get_platform_info(use_gpu=True)
slurm_args.update({
    'job_name': f'localretro_full_{TIMESTAMP}',
    'job_dir': str(PROJECT_ROOT / 'slurm' / 'jobs'),
    'output_dir': str(PROJECT_ROOT / 'slurm' / 'output'),
    'time': '48:00:00',
    'nodes': 1,
    'gpus-per-node': 1,
    'cpus-per-task': 8,
    'mem': '128G',
    'use_srun': True,
})

# --- Script commands: preprocessing then training ---
script_args = {
    'commands': [
        {
            'work_dir': str(PROJECT_ROOT / 'preprocessing'),
            'script': 'Extract_from_train_data.py',
            'args': {'-d': DATASET},
        },
        {
            'work_dir': str(PROJECT_ROOT / 'preprocessing'),
            'script': 'Run_preprocessing.py',
            'args': {'-d': DATASET},
        },
        {
            'work_dir': str(PROJECT_ROOT / 'scripts'),
            'script': 'Train.py',
            'args': {
                '-d': DATASET,
                '-b': 32,
                '-n': 50,
                '-nw': 4,
                '--overwrite': True,
            },
        },
    ],
}

os.makedirs(slurm_args['job_dir'], exist_ok=True)
os.makedirs(slurm_args['output_dir'], exist_ok=True)

create_and_submit_batch_job(slurm_args, script_args, interactive=slurm_args['interactive'])
