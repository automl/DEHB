#! /bin/bash

#SBATCH -p cluster-name
#SBATCH --gres=gpu:1
#SBATCH --mem 0
#SBATCH -J worker
#SBATCH -t 6-00

while getopts f:e:w: flag
do
    case "${flag}" in
        f) filename=${OPTARG};;    # specified as -f
        e) envname=${OPTARG};;     # specified as -e
        w) workername=${OPTARG};;  # specified as -w
    esac
done

# setting up environment
source $HOME/anaconda3/bin/activate $envname

# creating a Dask worker
PYTHONPATH=$PWD dask-worker --scheduler-file $filename --name $workername --resources "GPU=1" --no-nanny

# for more options: https://docs.dask.org/en/latest/setup/cli.html#dask-worker
