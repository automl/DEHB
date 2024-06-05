#!/bin/bash
#SBATCH -p 
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J DEHB_benchmarking              # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -t 0-10:00:00
#SBATCH --mem 16GB

BUDGET=300

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Benchmarking DEHB on multiple benchmarks";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    source activate dehb_pd1

    python benchmarking/mfpbench_benchmark.py --seed 0 --n_seeds 10 --fevals $BUDGET --benchmarks mfh3 mfh6 cifar100_wideresnet_2048 imagenet_resnet_512 lm1b_transformer_2048 translatewmt_xformer_64 --output_path logs/pd1

    python generate_summary.py
elif [ 2 -eq $SLURM_ARRAY_TASK_ID ]
then
    source activate dehb_jahs

    python benchmarking/mfpbench_benchmark.py --seed 0 --n_seeds 10 --fevals $BUDGET --benchmarks jahs --output_path logs/jahs

    python generate_summary.py
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    source activate dehb_hpo

    python benchmarking/hpobench_benchmark.py --seed 0 --n_seeds 10 --fevals $BUDGET --benchmarks tab_nn tab_rf tab_svm tab_lr surrogate nasbench201 --output_path logs/hpob

    python generate_summary.py
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    source activate dehb_pd1 # CountingOnes works with any dependencies, since it is only dependent on numpy

    python benchmarking/countingones_benchmark.py --seed 0 --n_seeds 10 --fevals $BUDGET --output_path logs/countingones --n_continuous 50 --n_categorical 50

    python generate_summary.py
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
