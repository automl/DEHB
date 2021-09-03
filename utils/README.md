### Scripts to setup a Dask cluster on Meta SLURM

There are 2 distinct ways in which DEHB can be run in a distributed manner.
* Letting the DEHB process create its [own Dask cluster](../README.md#running-dehb-in-a-parallel-setting) during runtime that lives and dies with the DEHB process 
* Setting up a Dask cluster that runs independently and multiple DEHB processes can connect and share the cluster

The scripts and instructions below account for the latter case, specifically for **SLURM**:

To create a Dask cluster with 10 workers and uses CPUs:
```bash
python utils/generate_slurm_jobs.py --worker_p [cpu_node] --scheduler_p [cpu_node] --nworkers 10 \
    --scheduler_path ./scheduler --scheduler_file scheduler_cpu.json --output_path temp --setup_file ./setup.sh
# generates 2 shell scripts
```
```bash
sbatch temp/scheduler.sh
# sleep 2s or wait till scheduler is allocated (not mandatory)
sbatch temp/workers.sh
```

Alternatively, to enable GPU usage by the workers,
```bash
python utils/generate_slurm_jobs.py --worker_p [cpu_node] --scheduler_p [cpu_node] --nworkers 10 \
    --scheduler_path ./scheduler --scheduler_file scheduler_gpu.json --output_path temp \
    --setup_file ./setup.sh --gpu
# generates 2 shell scripts
sbatch temp/scheduler.sh
# sleep 2s or wait till scheduler is allocated (not mandatory)
sbatch temp/workers.sh
```

The above sequence of commands will have a Dask cluster running and waiting for jobs. 
One or more DEHB processes can share this pool of 10 workers.
For example, running a DEHB optimization by specifiying `scheduler_file` makes that DEHB process, 
connect to the Dask cluster runnning.
```bash
python examples/03_pytorch_mnist_hpo.py --min_budget 1 --max_budget 9 --verbose \
    --scheduler_file scheduler/scheduler_gpu.json --runtime 200 --seed 123
```
The decoupled Dask cluster remains alive even after the DEHB optimization is over. 
It can be reused by other DEHB runs or processes. 
