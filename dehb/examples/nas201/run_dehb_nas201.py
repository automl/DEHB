'''Runs DEHB on NAS-Bench-201
'''

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas201/'))
sys.path.append(os.path.join(os.getcwd(), '../AutoDL-Projects/lib/'))

import json
import pickle
import argparse
import numpy as np
import ConfigSpace

from nas_201_api import NASBench201API as API
from models import CellStructure, get_search_spaces

from dehb import DE
from dehb import DEHB, DEHB_0, DEHB_1, DEHB_2, DEHB_3


# From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/BOHB.py
## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
def get_configuration_space(max_nodes, search_space):
  cs = ConfigSpace.ConfigurationSpace()
  #edge2index   = {}
  for i in range(1, max_nodes):
    for j in range(i):
      node_str = '{:}<-{:}'.format(i, j)
      cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(node_str, search_space))
  return cs


# From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/BOHB.py
## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
def config2structure_func(max_nodes):
  def config2structure(config):
    genotypes = []
    for i in range(1, max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        op_name = config[node_str]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return CellStructure( genotypes )
  return config2structure


def calculate_regrets(history, runtime):
    assert len(runtime) == len(history)
    global dataset, api, de, max_budget

    regret_test = []
    regret_validation = []
    inc = np.inf
    test_regret = 1
    validation_regret = 1
    for i in range(len(history)):
        config, valid_regret, budget = history[i]
        valid_regret = valid_regret - y_star_valid
        if valid_regret <= inc:
            inc = valid_regret
            config = de.vector_to_configspace(config)
            structure = config2structure(config)
            arch_index = api.query_index_by_arch(structure)
            info = api.get_more_info(arch_index, dataset, max_budget, False, False)
            test_regret = (1 - (info['test-accuracy'] / 100)) - y_star_test
        regret_validation.append(inc)
        regret_test.append(test_regret)
    res = {}
    res['regret_test'] = regret_test
    res['regret_validation'] = regret_validation
    res['runtime'] = np.cumsum(runtime).tolist()
    return res


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(output_path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


def find_nas201_best(api, dataset):
    arch, y_star_test = api.find_best(dataset=dataset, metric_on_set='ori-test')
    _, y_star_valid = api.find_best(dataset=dataset, metric_on_set='x-valid')
    return 1 - (y_star_valid / 100), 1 - (y_star_test / 100)


parser = argparse.ArgumentParser()
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                    nargs='?', help='seed')
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--dataset', default='cifar10-valid', type=str, nargs='?',
                    choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'],
                    help='choose the dataset')
parser.add_argument('--max_nodes', default=4, type=int, nargs='?',
                    help='maximum number of nodes in the cell')
parser.add_argument('--iter', default=100, type=int, nargs='?',
                    help='number of DEHB iterations')
parser.add_argument('--gens', default=1, type=int, nargs='?',
                    help='number of generations for DE to evolve')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', type=str, nargs='?',
                    default="../nas201/NAS-Bench-201-v1_1-096897.pth",
                    help='specifies the path to the benchmark data')
strategy_choices = ['rand1_bin', 'rand2_bin', 'rand2dir_bin', 'best1_bin', 'best2_bin',
                    'currenttobest1_bin', 'randtobest1_bin',
                    'rand1_exp', 'rand2_exp', 'rand2dir_exp', 'best1_exp', 'best2_exp',
                    'currenttobest1_exp', 'randtobest1_exp']
parser.add_argument('--strategy', default="rand1_bin", choices=strategy_choices,
                    type=str, nargs='?',
                    help="specify the DE strategy from among {}".format(strategy_choices))
parser.add_argument('--mutation_factor', default=0.5, type=float, nargs='?',
                    help='mutation factor value')
parser.add_argument('--crossover_prob', default=0.5, type=float, nargs='?',
                    help='probability of crossover')
parser.add_argument('--min_budget', default=11, type=int, nargs='?',
                    help='minimum budget')
parser.add_argument('--max_budget', default=199, type=int, nargs='?',
                    help='maximum budget')
parser.add_argument('--eta', default=3, type=int, nargs='?',
                    help='hyperband eta')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default=None, type=str, nargs='?',
                    help='name of folder where files will be dumped')
parser.add_argument('--version', default=None, type=str, nargs='?',
                    help='DEHB version to run')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False
min_budget = args.min_budget
max_budget = args.max_budget
dataset = args.dataset

# Directory where files will be written
if args.folder is None:
    folder = "dehb_v{}".format(args.version)
else:
    folder = args.folder

output_path = os.path.join(args.output_path, args.dataset, folder)
os.makedirs(output_path, exist_ok=True)

# Loading NAS-201
api = API(args.data_dir)
search_space = get_search_spaces('cell', 'nas-bench-201')

# Parameter space to be used by DE
cs = get_configuration_space(args.max_nodes, search_space)
dimensions = len(cs.get_hyperparameters())
config2structure = config2structure_func(args.max_nodes)

y_star_valid, y_star_test = find_nas201_best(api, dataset)
inc_config = cs.get_default_configuration().get_array().tolist()


# Custom objective function for DEHB to interface NASBench-201
def f(config, budget=max_budget):
    global dataset, api
    structure = config2structure(config)
    arch_index = api.query_index_by_arch(structure)
    if budget is not None:
        budget = int(budget)
    # From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/R_EA.py
    ## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
    info = api.get_more_info(arch_index, dataset, iepoch=budget,
                             use_12epochs_result=False, is_random=True)
    try:
        fitness = info['valid-accuracy']
    except:
        fitness = info['valtest-accuracy']

    cost = info['train-all-time']
    try:
        cost += info['valid-all-time']
    except:
        cost += info['valtest-all-time']

    fitness = 1 - fitness / 100
    return fitness, cost


dehbs = {None: DEHB, "0": DEHB_0, "1": DEHB_1, "2": DEHB_2, "3": DEHB_3}
DEHB = dehbs[args.version]

# Initializing DEHB object
dehb = DEHB(cs=cs, dimensions=dimensions, f=f, strategy=args.strategy,
            mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
            eta=args.eta, min_budget=min_budget, max_budget=max_budget,
            generations=args.gens)
# Initializing DE object
de = DE(cs=cs, dimensions=dimensions, f=f, pop_size=10,
        mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
        strategy=args.strategy, budget=args.max_budget)

if args.runs is None:  # for a single run
    if not args.fix_seed:
        np.random.seed(0)
    # Running DE iterations
    traj, runtime, history = dehb.run(iterations=args.iter, verbose=args.verbose)
    res = calculate_regrets(history, runtime)
    fh = open(os.path.join(output_path, 'run_{}.json'.format(args.run_id)), 'w')
    json.dump(res, fh)
    fh.close()
else:  # for multiple runs
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if not args.fix_seed:
            np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        # Running DE iterations
        traj, runtime, history = dehb.run(iterations=args.iter, verbose=args.verbose)
        res = calculate_regrets(history, runtime)
        fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
        json.dump(res, fh)
        fh.close()
        print("Run saved. Resetting...")
        # essential step to not accumulate consecutive runs
        dehb.reset()

save_configspace(cs, output_path)
