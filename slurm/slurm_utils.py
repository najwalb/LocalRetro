
import os
import subprocess
from datetime import datetime
from pathlib import Path
import argparse

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]


def get_platform_info(use_gpu=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--platform', type=str)
    args = parser.parse_args()
    platform = args.platform

    if platform == 'puhti':
        project = 'project_2015643'
        partition = 'gpu' if use_gpu else 'small'
        puhti_module = 'pytorch/2.4'
        venv_path = '/projappl/project_2015643/localretro'
    elif platform == 'mahti':
        project = 'project_2015643'
        partition = 'gpusmall' if use_gpu else 'small'
        puhti_module = 'pytorch/2.4'
        venv_path = '/projappl/project_2015643/localretro'
    elif platform == 'lumi':
        project = 'project_462001028'
        partition = 'small-g' if use_gpu else 'small'
        puhti_module = 'pytorch/2.4'
        venv_path = '/projappl/project_462001028/localretro'
    else:
        raise ValueError(f'Platform {platform} not supported')
    return {
        'platform': platform,
        'project': project,
        'partition': partition,
        'venv_path': venv_path,
        'puhti_module': puhti_module,
        'interactive': args.interactive,
    }


def add_general_slurm_job_setup(fh, slurm_args):
    fh.writelines("#!/bin/bash\n")
    fh.writelines(f"#SBATCH --job-name={slurm_args['job_name']}\n")
    fh.writelines(f"#SBATCH --account={slurm_args['project']}\n")
    fh.writelines(f"#SBATCH --partition={slurm_args['partition']}\n")
    fh.writelines(f"#SBATCH --output={slurm_args['output_dir']}/{slurm_args['job_name']}_%j.out\n")
    fh.writelines(f"#SBATCH --error={slurm_args['output_dir']}/{slurm_args['job_name']}_%j.err\n")
    fh.writelines(f"#SBATCH --time={slurm_args['time']}\n")
    if 'dependency' in slurm_args:
        fh.writelines(f"#SBATCH --dependency=afterok:{slurm_args['dependency']}\n")


def add_platform_specific_slurm_commands(fh, slurm_args):
    if slurm_args['platform'] == 'lumi':
        fh.writelines(f"#SBATCH --nodes={slurm_args['nodes']}\n")
        if 'gpus-per-node' in slurm_args and slurm_args['gpus-per-node'] > 0:
            fh.writelines(f"#SBATCH --gpus-per-node={slurm_args['gpus-per-node']}\n")
        fh.writelines(f"#SBATCH --ntasks-per-node={slurm_args.get('ntasks-per-node', 1)}\n")
        fh.writelines(f"#SBATCH --cpus-per-task={slurm_args['cpus-per-task']}\n")
        fh.writelines(f"#SBATCH --mem={slurm_args['mem']}\n\n")
        fh.writelines('module use /appl/local/csc/modulefiles/\n')
        fh.writelines(f'module load {slurm_args["puhti_module"]}\n')
        fh.writelines(f"export PYTHONUSERBASE={slurm_args['venv_path']}\n\n")
    elif slurm_args['platform'] in ('puhti', 'mahti'):
        fh.writelines(f"#SBATCH --nodes={slurm_args['nodes']}\n")
        if slurm_args['partition'] in ('gpu', 'gputest', 'gpusmall', 'gpumedium'):
            if slurm_args['platform'] == 'mahti':
                fh.writelines(f"#SBATCH --gres=gpu:a100:{slurm_args['gpus-per-node']}\n")
            else:
                fh.writelines(f"#SBATCH --gres=gpu:v100:{slurm_args['gpus-per-node']}\n")
        fh.writelines(f"#SBATCH --cpus-per-task={slurm_args['cpus-per-task']}\n")
        fh.writelines(f"#SBATCH --mem={slurm_args['mem']}\n\n")
        fh.writelines("module purge\n")
        fh.writelines(f"module load {slurm_args['puhti_module']}\n")
        fh.writelines(f"export PYTHONUSERBASE={slurm_args['venv_path']}\n\n")
    else:
        raise ValueError(f"Platform {slurm_args['platform']} not supported")


def add_script_commands(fh, slurm_args, script_args):
    """Write the script execution commands into the SLURM job file.

    For LocalRetro, arguments use argparse format (--key value) not Hydra (key=value).
    """
    os.makedirs(slurm_args['job_dir'], exist_ok=True)
    job_file = os.path.join(slurm_args['job_dir'],
                            f"{slurm_args['job_name']}.sh")

    with open(job_file, 'w') as fj:
        fj.write("#!/bin/bash\nset -e\n")
        fj.write(f"export PYTHONUSERBASE={slurm_args['venv_path']}\n\n")
        # Write each command
        for cmd in script_args['commands']:
            fj.write(f"cd {cmd['work_dir']}\n")
            line = f"python {cmd['script']}"
            for arg, value in cmd.get('args', {}).items():
                if isinstance(value, bool) and value:
                    line += f" {arg}"
                else:
                    line += f" {arg} {value}"
            fj.write(line + "\n\n")

    if fh is not None:
        fh.write(f"chmod +x {job_file}\n")
        if slurm_args.get('use_srun', False):
            fh.write(f"srun bash {job_file}\n")
        else:
            fh.write(f"bash {job_file}\n")

    return job_file


def create_and_submit_batch_job(slurm_args, script_args, interactive=False):
    if interactive:
        # Run locally
        job_file = add_script_commands(None, slurm_args, script_args)
        print(f"Running script interactively: {job_file}")
        result = subprocess.Popen(["bash", job_file],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        stdout, stderr = result.communicate()
        print(stdout.decode())
        if stderr:
            print(stderr.decode())
    else:
        print(f"Creating job file for {slurm_args['job_name']} in {slurm_args['job_dir']}")
        os.makedirs(slurm_args['job_dir'], exist_ok=True)
        job_file = os.path.join(slurm_args['job_dir'],
                                f"{slurm_args['job_name']}.job")
        with open(job_file, 'w') as fh:
            add_general_slurm_job_setup(fh, slurm_args)
            add_platform_specific_slurm_commands(fh, slurm_args)
            add_script_commands(fh, slurm_args, script_args)

        print(f"Submitting {job_file}")
        result = subprocess.Popen(["/usr/bin/sbatch", job_file],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        stdout, stderr = result.communicate()
        output = stdout.decode("utf-8")
        if 'job' not in output:
            print(f"Error: {stderr.decode()}")
        else:
            job_id = output.strip().split('job ')[1]
            print(f"=== {slurm_args['job_name']}. Slurm ID = {job_id}.")
            return job_id
