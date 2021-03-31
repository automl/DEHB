#! /bin/bash

while getopts f:e:n: flag
do
    case "${flag}" in
        f) filename=${OPTARG};;  # specified as -f
        e) envname=${OPTARG};;   # specified as -e
        n) nworkers=${OPTARG};;  # specified as -n
    esac
done

echo "Submitting Dask scheduler..."
sbatch utils/dask_scheduler.sh -f $filename -e $envname

for ((i=1; i<=$nworkers; i++)); do
   echo "Submitting worker "$i"..."
   sbatch utils/dask_workers.sh -f $filename -e $envname -w worker$i
   sleep 2
done
