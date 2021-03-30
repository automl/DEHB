#! /bin/bash

#SBATCH -p mldlc_gpu-rtx2080
#SBATCH --gres=gpu:0
#SBATCH --mem 1G
#SBATCH -c 2
#SBATCH -J scheduler
#SBATCH -t 6-00

while getopts f:e: flag
do
    case "${flag}" in
        f) filename=${OPTARG};;  # specified as -f
        e) envname=${OPTARG};;   # specified as -e
    esac
done

# setting up environment
source $HOME/anaconda3/bin/activate $envname

# Creating a Dask scheduler
PYTHONPATH=$PWD dask-scheduler --scheduler-file $filename

# for more options: https://docs.dask.org/en/latest/setup/cli.html#dask-scheduler
