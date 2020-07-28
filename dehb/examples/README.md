To install the required HPOlib2 for benchmarks:

`pip install git+https://github.com/automl/HPOlib2.git@development`


To install HpBandSter:

`pip install hpbandster`


SciPy needed: 1.2.1


Cartpole needs: Tensorforce 0.4.4


BNN needs: https://github.com/automl/sgmcmc.git



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
└───HpBandSter/   
|   └───icml_2018_experiments/
│   |   └───experiments/
│   |   |   └───workers/
|   └───...
```
