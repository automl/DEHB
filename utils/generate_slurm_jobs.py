"""
Generates 2 scripts to deploy a Dask cluster to SLURM.

Example use:
```
python utils/generate_slurm_jobs.py --worker_p [gpu_node] --scheduler_p [cpu_node] --gpu \
    --nworkers 4 --scheduler_path ./scheduler --output_path temp --setup_file ./setup.sh`
```

Generated files can be submitted:
```
sbatch temp/scheduler.sh
sbatch temp/workers.sh
```
"""

import os
import argparse
from pathlib import Path


def scheduler_command(scheduler_file):
    cmd = "dask-scheduler --scheduler-file {}"
    cmd = cmd.format(scheduler_file)
    cmd += "\n"
    return cmd


def worker_command(scheduler_file, worker_name, gpu=False, gpu_per_worker=1):
    cmd = "dask-worker --scheduler-file {} --name \"{}_\"$SLURM_ARRAY_TASK_ID --no-nanny"
    extra_args = " --reconnect --nprocs 1 --nthreads 1"
    cmd = cmd.format(scheduler_file, worker_name)
    if gpu:
        cmd += " --resources \"GPU={}\"".format(gpu_per_worker)
    cmd += extra_args
    cmd += "\n"
    return cmd


def slurm_header(args, worker=False):
    cmds = list()
    # adding shebang
    cmds.append("#! /bin/bash")
    node = args.worker_p if worker else args.scheduler_p
    # adding target node
    cmds.append("#SBATCH -p {}".format(node))
    if not worker:
        # adding cpu request
        cmds.append("#SBATCH -c {}".format(args.c))
    # adding timelimit
    cmds.append("#SBATCH -t {}".format(args.t))
    # adding job name
    suffix = "worker" if worker else "scheduler"
    cmds.append("#SBATCH -J {}-{}".format(args.J, suffix))
    if args.gpu and worker:
        # adding gpu request
        cmds.append("#SBATCH --gres=gpu:{}".format(args.gpu_per_worker))
        # making an array job for the workers
        cmds.append("#SBATCH -a 1-{}".format(args.nworkers))
    log_pattern = str(args.slurm_dump_path / "slurm_%j-%a-%x.{}")
    # adding error directory
    cmds.append("#SBATCH -e {}".format(log_pattern.format("err")))
    cmds.append("#SBATCH -o {}".format(log_pattern.format("out")))
    cmds.append("\n")
    cmds = "\n".join(cmds)
    return cmds


def input_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--scheduler_file",
        default="scheduler.json",
        type=str,
        help="The file name storing the Dask cluster connections"
    )
    parser.add_argument(
        "--scheduler_path",
        default="./scheduler",
        type=str,
        help="The path to keep the scheduler.json like files for Dask"
    )
    parser.add_argument(
        "--setup_file",
        default=None,
        type=str,
        help="The path to file that will be sourced to load environment and set path variables"
    )
    parser.add_argument(
        "--output_path", default="./", type=str, help="The path to dump the generated script"
    )
    parser.add_argument(
        "--slurm_dump_path", default="./slurm-logs", type=str, help="Path to dump the slurm logs"
    )
    parser.add_argument(
        "--nworkers", default=10, type=int, help="Number of workers to run"
    )
    parser.add_argument(
        "--worker_name", default="w", type=str, help="Dask worker name prefix"
    )
    parser.add_argument(
        "-c", default=2, type=int, help="CPUs per task requested"
    )
    parser.add_argument(
        "--gpu", default=False, action="store_true", help="If set, the workers request GPUs"
    )
    parser.add_argument(
        "--gpu_per_worker", default=1, type=int, help="Number of GPUs per worker"
    )
    parser.add_argument(
        "--scheduler_p", default=None, required=True, type=str, help="The node to submit schedulers"
    )
    parser.add_argument(
        "--worker_p", default=None, required=True, type=str, help="The node to submit workers"
    )
    parser.add_argument(
        "-t", default="1:00:00", type=str, help="TIMELIMIT"
    )
    parser.add_argument(
        "-J", default="dehb", type=str, help="Prefix to scheduler and worker job names"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_arguments()

    args.slurm_dump_path = Path(args.slurm_dump_path).absolute()
    scheduler = Path(args.scheduler_path).absolute() / args.scheduler_file
    os.makedirs(Path(args.scheduler_path).absolute(), exist_ok=True)
    output_path = Path(args.output_path).absolute()
    os.makedirs(Path(args.output_path).absolute(), exist_ok=True)
    scheduler_file = output_path / "scheduler.sh"
    worker_file = output_path / "workers.sh"
    setup_cmd = "source {}\n\n".format(Path(args.setup_file).absolute())

    # generating scheduler script
    cmd = slurm_header(args, worker=False)
    cmd += setup_cmd
    cmd += scheduler_command(scheduler_file=scheduler)
    cmd += "\n"
    with open(scheduler_file, "w") as f:
        f.writelines(cmd)
    print("Saving scheduler job script to {}".format(scheduler_file))
    # generating worker script
    cmd = slurm_header(args, worker=True)
    cmd += setup_cmd
    cmd += worker_command(
        scheduler_file=scheduler,
        worker_name=args.worker_name,
        gpu=args.gpu,
        gpu_per_worker=args.gpu_per_worker
    )
    with open(worker_file, "w") as f:
        f.writelines(cmd)
    print("Saving worker job script to {}".format(worker_file))
