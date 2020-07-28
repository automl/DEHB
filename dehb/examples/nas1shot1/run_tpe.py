# Slightly modified version of the script from:
# https://github.com/automl/nasbench-1shot1/blob/master/optimizers/tpe/run_tpe.py

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nasbench/'))
sys.path.append(os.path.join(os.getcwd(), '../nasbench-1shot1/'))

import pickle
import argparse
import logging
import numpy as np
import ConfigSpace
logging.basicConfig(level=logging.INFO)

from copy import deepcopy
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from nasbench import api

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from nasbench_analysis.utils import INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3

from IPython import embed


parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--search_space', default=None, type=str, nargs='?',
                    help='specifies the benchmark')
parser.add_argument('--n_iters', default=280, type=int, nargs='?',
                    help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./experiments", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', type=str, nargs='?',
                    default="../nasbench-1shot1/nasbench_analysis/nasbench_data/"
                            "108_e/nasbench_full.tfrecord",
                    help='specifies the path to the nasbench data')
parser.add_argument('--seed', default=0, type=int,
                    help='random seed')
parser.add_argument('--n_repetitions', default=500, type=int,
                    help='number of repetitions')
args = parser.parse_args()

nasbench = api.NASBench(args.data_dir)

output_path = os.path.join(args.output_path, "discrete_optimizers", 'TPE')
os.makedirs(os.path.join(output_path), exist_ok=True)

if args.search_space is None:
    spaces = [1, 2, 3]
else:
    spaces = [int(args.search_space)]

#embed()

def objective_function(config):
    config_copy = deepcopy(config)
    c = ConfigSpace.Configuration(cs, values=config_copy)
    y, cost = search_space.objective_function(nasbench, c, budget=108)
    return {
        'config': config_copy,
        'loss': 1 - float(y),
        'cost': cost,
        'status': STATUS_OK
    }

for space in spaces:
    print('##### Search Space {} #####'.format(space))
    search_space = eval('SearchSpace{}()'.format(space))
    cs = search_space.get_configuration_space()

    hyperopt_space = {h.name: hp.choice(h.name, h.choices) for h in
                      cs.get_hyperparameters()}

    #for seed in range(args.n_repetitions):
    print('##### Seed {} #####'.format(args.seed))
    # Set random_seed
    np.random.seed(args.seed)

    trials = Trials()
    best = fmin(objective_function,
                space=hyperopt_space,
                algo=tpe.suggest,
                max_evals=args.n_iters,
                trials=trials)

    fh = open(os.path.join(output_path,
                           'algo_{}_{}_ssp_{}_seed_{}.obj'.format('TPE',
                                                                  args.run_id,
                                                                  space,
                                                                  args.seed)), 'wb')
    pickle.dump(search_space.run_history, fh)
    fh.close()

    print(min([1 - arch.test_accuracy - search_space.test_min_error for
               arch in search_space.run_history]))
