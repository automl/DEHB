### DEHB experiments

To install the required __HPOlib2__ for benchmarks:

`pip install git+https://github.com/automl/HPOlib2.git@development`


To install __HpBandSter__:

`pip install hpbandster`


__SciPy__ needed: 1.2.1


__Cartpole__ needs: Tensorforce 0.4.4


__BNN__ needs: https://github.com/automl/sgmcmc.git


Requirements for running experiments with __NAS benchmarks__:

* __nas\_benchmarks__ can be found [here](https://github.com/automl/nas_benchmarks/blob/development/README.md)
along with the instructions to download the files
    * This repo contains both NAS-Bench-101 and NAS-HPO-Bench
* __nas-bench-1shot1__ can be found [here](https://github.com/automl/nasbench-1shot1)
* __AutoDL-Projects__ can be found [here](https://github.com/D-X-Y/AutoDL-Projects)
    * This repo contains NAS-201 and the benchmark files can be found 
    [here](https://github.com/D-X-Y/NAS-Bench-201#preparation-and-download)
* __nasbench__ can be found [here](https://github.com/google-research/nasbench)
<br/>
<br/>

Directory structure for the execution of these scripts:
```
..    
|
└───DEHB/   
│   └───dehb/
|
└───nas_benchmarks/
│   └───experiment_scripts/
│   └───tabular_benchmarks/
|   |   └───fcnet_benchmark.py
|   |   └───nas_cifar10.py
|   |   └───fcnet_tabular_benchmarks/
|   |   |   └───nasbench_full.tfrecord
|   |   |   └───nasbench_only108.tfrecord
|   |   |   └───fcnet_naval_propulsion_data.hdf5
|   |   |   └───fcnet_protein_structure_data.hdf5
|   |   |   └───fcnet_slice_localization_data.hdf5
|   |   |   └───fcnet_parkinsons_telemonitoring_data.hdf5
|   └───...
|
└───nasbench-1shot1/
│   └───nasbench_analysis/
│   |   └───nasbench_data/   
|   |   │   └───108_e/
│   |   |       └───nasbench_full.tfrecord
│   |   |       └───nasbench_only108.tfrecord
|   └───optimizers/
|   └───...
|
└───nasbench/
│   └───lib/
|   └───api.py/
|   └───...
|
└───AutoDL-Projects/
│   └───exps/
|   └───libs/
|   └───...
|
└───nas201/
│   └───NAS-Bench-201-v1_1-096897.pth
```