# Slightly modified version of the script from:
# https://github.com/automl/nasbench-1shot1/blob/master/optimizers/smac/run_smac.py

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nasbench/'))
sys.path.append(os.path.join(os.getcwd(), '../nasbench-1shot1/'))

import json
import pickle
import argparse
import logging
import numpy as np
import ConfigSpace
logging.basicConfig(level=logging.INFO)

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO as SMAC
from multiprocessing.managers import BaseManager

from nasbench import api

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from nasbench_analysis.utils import INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3


def objective_function(config, **kwargs):
    global search_space, nasbench
    fitness, cost = search_space.objective_function(nasbench, config, budget=108)
    fitness = 1 - fitness
    return float(fitness)


def benchmark_wrapper(space, *args, **kwargs):
    if space == 1:
        obj = SearchSpace1(*args, **kwargs)
    elif space == 2:
        obj = SearchSpace2(*args, **kwargs)
    else:
        obj = SearchSpace3(*args, **kwargs)
    # the AutoProxy wrapper around search_space, created by BaseManager doesn't expose attributes
    # the lambda function below creates a function that can be used to access attributes
    # obj.get_run_history() becomes equivalent to obj.run_history
    obj.get_run_history = lambda: obj.run_history
    obj.get_valid_min_error = lambda: obj.valid_min_error
    obj.get_test_min_error = lambda: obj.test_min_error
    return obj


parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--search_space', default=None, type=str, nargs='?',
                    help='specifies the benchmark')
parser.add_argument('--n_iters', default=10, type=int, nargs='?',
                    help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./experiments", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir',
                    default="../nasbench-1shot1/nasbench_analysis/nasbench_data/108_e/nasbench_only108.tfrecord",
                    type=str, nargs='?', help='specifies the path to the nasbench data')
parser.add_argument('--seed', default=0, type=int,
                    help='random seed')
args = parser.parse_args()

nasbench = api.NASBench(args.data_dir)

output_path = os.path.join(args.output_path, "SMAC")
os.makedirs(os.path.join(output_path), exist_ok=True)

BaseManager.register('benchmark', benchmark_wrapper)
manager = BaseManager()

if args.search_space is None:
    spaces = [1, 2, 3]
else:
    spaces = [int(args.search_space)]

for space in spaces:
    print('##### Search Space {} #####'.format(space))
    print('##### Seed {} #####'.format(args.seed))
    np.random.seed(args.seed)
    run_id = args.run_id

    # important to let SMAC workers access the shared object
    manager.start()
    search_space = manager.benchmark(space=space)

    cs = search_space.get_configuration_space()
    y_star_valid, y_star_test, inc_config = (search_space.get_valid_min_error(),
                                             search_space.get_test_min_error(), None)

    scenario = Scenario({"run_obj": "quality",
                         "runcount-limit": args.n_iters,
                         "cs": cs,
                         "deterministic": "false",
                         "initial_incumbent": "RANDOM",
                         "output_dir": ""})
    smac = SMAC(scenario=scenario, tae_runner=objective_function)
    smac.optimize()
    # collects the optimization trace from the search_space object which kept track of evaluations
    run_history = search_space.get_run_history()
    # important to reset the tracker or re-initialize the search_space object
    manager.shutdown()
    fh = open(os.path.join(output_path,
                           'SMAC_{}_ssp_{}_seed_{}.obj'.format(run_id, space, run_id)), 'wb')
    pickle.dump(run_history, fh)
    fh.close()
