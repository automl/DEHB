### Running DEHB in a parallel setting

DEHB has been designed to interface a [Dask client](https://distributed.dask.org/en/latest/api.html#distributed.Client).
DEHB can either create a Dask client during instantiation and close/kill the client during garbage collection. 
Or a client can be passed as an argument during instantiation.

* Setting `n_workers` during instantiation  
    If set to `1` (default) then the entire process is a sequential run without invoking Dask.  
    If set to `>1` then a Dask Client is initialized with as many workers as `n_workers`.  
    This parameter is ignored if `client` is not None.
* Setting `client` during instantiation  
    When `None` (default), a Dask client is created using `n_workers` specified.  
    Else, any custom-configured Dask Client can be created and passed as the `client` argument to DEHB.
  
#### Using GPUs in a parallel run

Certain target function evaluations (especially for Deep Learning) require computations to be 
carried out on GPUs. The GPU devices are often ordered by device ID and if not configured, all 
spawned worker processes access these devices in the same order and can either run out of memory or
not exhibit parallelism.

For `n_workers>1` and when running on a single node (or local), the `single_node_with_gpus` can be 
passed to the `run()` call to DEHB. Setting it to `False` (default) has no effect on the default setup 
of the machine. Setting it to `True` will reorder the GPU device IDs dynamically by setting the environment 
variable `CUDA_VISIBLE_DEVICES` for each worker process executing a target function evaluation. The re-ordering 
is done in a manner that the first priority device is the one with the least number of active jobs assigned 
to it by that DEHB run.

To run the PyTorch MNIST example on a single node using 2 workers:  
```bash
python examples/03_pytorch_mnist_hpo.py \
    --min_budget 1 \
    --max_budget 3 \
    --runtime 60 \
    --n_workers 2 \
    --single_node_with_gpus \
    --verbose
```

#### Multi-node runs

Multi-node parallelism is often contingent on the cluster setup to be deployed on. Dask provides useful 
frameworks to interface various cluster designs. As long as the `client` passed to DEHB during 
instantiation is of type `dask.distributed.Client`, DEHB can interact with this client and 
distribute its optimization process in a parallel manner. 

For instance, `Dask-CLI` can be used to create a `dask-scheduler` which can dump its connection 
details to a file on a cluster node accessible to all processes. Multiple `dask-worker` can then be
created to interface the `dask-scheduler` by connecting to the details read from the file dumped. Each
dask-worker can be triggered on any remote machine. Each worker can be configured as required, 
including mapping to specific GPU devices. 

Some helper scripts can be found [here](https://github.com/automl/DEHB/tree/master/utils), that can be used as a reference to run DEHB in a multi-node 
manner on clusters managed by SLURM. (*not expected to work off-the-shelf*)

To run the PyTorch MNIST example on a multi-node setup using 4 workers:
```bash
bash utils/run_dask_setup.sh \
    -n 4 \
    -f dask_dump/scheduler.json \   # This is how the workers will be discovered by DEHB
    -e env_name

# Make sure to sleep to allow the workers to setup properly
sleep 5

python examples/03_pytorch_mnist_hpo.py \
    --min_fidelity 1 \
    --max_fidelity 3 \
    --runtime 60 \
    --scheduler_file dask_dump/scheduler.json \
    --verbose
```