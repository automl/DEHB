# Slightly modified version of the script from:
# https://github.com/automl/nasbench-1shot1/blob/master/optimizers/smac/run_smac.py

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

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncDict
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
parser.add_argument('--random_fraction', default=.33, type=float, nargs='?',
                    help='fraction of random configurations')
parser.add_argument('--n_trees', default=10, type=int, nargs='?',
                    help='number of trees for the random forest')
parser.add_argument('--max_feval', default=4, type=int, nargs='?',
                    help='maximum number of function evaluation per configuration')
parser.add_argument('--output_path', default="./experiments", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir',
                    default="nasbench_analysis/nasbench_data/108_e/nasbench_only108.tfrecord",
                    type=str, nargs='?', help='specifies the path to the nasbench data')
parser.add_argument('--seed', default=0, type=int,
                    help='random seed')
parser.add_argument('--n_repetitions', default=500, type=int,
                    help='number of repetitions')
args = parser.parse_args()

nasbench = api.NASBench(args.data_dir)

output_path = os.path.join(args.output_path, "discrete_optimizers", 'SMAC')
os.makedirs(os.path.join(output_path), exist_ok=True)

if args.search_space is None:
    spaces = [1, 2, 3]
else:
    spaces = [int(args.search_space)]

#embed()

def objective_function(config, **kwargs):
    c = ConfigSpace.Configuration(cs, values=config)
    y, cost = search_space.objective_function(nasbench, c, budget=108)
    return 1 - float(y)

for space in spaces:
    print('##### Search Space {} #####'.format(space))
    search_space = eval('SearchSpace{}()'.format(space))
    cs = search_space.get_configuration_space()

    #for seed in range(args.n_repetitions):
    print('##### Seed {} #####'.format(args.seed))
    # Set random_seed
    np.random.seed(args.seed)

    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": args.n_iters,
        "cs": cs,
        "deterministic": "false",
        "initial_incumbent": "RANDOM",
        "output_dir": output_path
    })

    tae = ExecuteTAFuncDict(objective_function, use_pynisher=False)
    smac = SMAC(scenario=scenario, tae_runner=tae)

    # probability for random configurations
    smac.solver.random_configuration_chooser.prob = args.random_fraction
    smac.solver.model.rf_opts.num_trees = args.n_trees
    # only 1 configuration per SMBO iteration
    smac.solver.scenario.intensification_percentage = 1e-10
    smac.solver.intensifier.min_chall = 1
    # maximum number of function evaluations per configuration
    smac.solver.intensifier.maxR = args.max_feval

    smac.optimize()

    fh = open(os.path.join(output_path,
                           'algo_{}_{}_ssp_{}_seed_{}.obj'.format('SMAC',
                                                                  args.run_id,
                                                                  space,
                                                                  args.seed)), 'wb')
    pickle.dump(search_space.run_history, fh)
    fh.close()

    print(min([1 - arch.test_accuracy - search_space.test_min_error for
               arch in search_space.run_history]))