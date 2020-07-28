#########################
# Tree Parzen Estimator #
#########################
# This script is a modified version of the script authored by Aaron Klein
# https://github.com/automl/nas_benchmarks/blob/development/experiment_scripts/run_tpe.py

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas201/'))
sys.path.append(os.path.join(os.getcwd(), '../AutoDL-Projects/lib/'))


import os
import time
import json
import random
import pickle
import logging
import argparse
import collections
import numpy as np
import ConfigSpace
from copy import deepcopy

logging.basicConfig(level=logging.ERROR)

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from nas_201_api import NASBench201API as API
from models import CellStructure, get_search_spaces


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

def find_nas201_best(api, dataset):
    arch, y_star_test = api.find_best(dataset=dataset, metric_on_set='ori-test')
    _, y_star_valid = api.find_best(dataset=dataset, metric_on_set='x-valid')
    return 1 - (y_star_valid / 100), 1 - (y_star_test / 100)

# Custom objective function to interface Nasbench-201
def f(config, budget=199):
    global nas_bench, dataset
    structure = config2structure(config)
    arch_index = nas_bench.query_index_by_arch(structure)
    if budget is not None:
      budget = int(budget)
    # From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/R_EA.py
    ## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
    info = nas_bench.get_more_info(arch_index, dataset, iepoch=budget,
                                   use_12epochs_result=False, is_random=True)
    try:
      val_score = info['valid-accuracy']
    except:
      val_score = info['valtest-accuracy']

    cost = info['train-all-time']
    try:
      cost += info['valid-all-time']
    except:
      cost += info['valtest-all-time']

    val_score = 1 - val_score / 100

    info = nas_bench.get_more_info(arch_index, dataset, iepoch=max_budget,
                                   use_12epochs_result=False, is_random=False)
    test_score = 1 - info['test-accuracy'] / 100

    return val_score, cost, test_score


def convert_to_json(results):
    global y_star_valid, y_star_test
    regret_validation = []
    regret_test = []
    runtime = []
    inc = np.inf
    test_regret = 1
    validation_regret = 1

    for i in range(len(results)):
        validation_regret = results[i]['loss'] - y_star_valid
        if validation_regret <= inc:
            inc = validation_regret
            test_regret = results[i]['test'] - y_star_test
        regret_validation.append(inc)
        regret_test.append(test_regret)
        runtime.append(results[i]['cost'])

    res = {}
    res['regret_validation'] = regret_validation
    res['regret_test'] = regret_test
    res['runtime'] = np.cumsum(runtime).tolist()
    return res


parser = argparse.ArgumentParser()
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                    nargs='?', help='seed')
parser.add_argument('--dataset', default='cifar10-valid', type=str, nargs='?',
                    choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'],
                    help='choose the dataset')
parser.add_argument('--max_nodes', default=4, type=int, nargs='?',
                    help='maximum number of nodes in the cell')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--n_iters', default=10, type=int, nargs='?',
                    help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results/", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../nas201/NAS-Bench-201-v1_1-096897.pth",
                    type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--min_budget', default=11, type=int, nargs='?',
                    help='minimum budget for BOHB')
parser.add_argument('--max_budget', default=199, type=int, nargs='?',
                    help='maximum budget for BOHB')
parser.add_argument('--folder', default='tpe', type=str, nargs='?',
                    help='name of folder where files will be dumped')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')

args = parser.parse_args()
dataset = args.dataset
max_budget = args.max_budget
min_budget = args.min_budget

output_path = os.path.join(args.output_path, args.dataset, args.folder)
os.makedirs(output_path, exist_ok=True)

# Loading NAS-201
nas_bench = API(args.data_dir)
search_space = get_search_spaces('cell', 'nas-bench-201')

# Getting configuration space
cs = get_configuration_space(args.max_nodes, search_space)
dimensions = len(cs.get_hyperparameters())
config2structure = config2structure_func(args.max_nodes)

space = {}
for h in cs.get_hyperparameters():
    if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
        space[h.name] = hp.quniform(h.name, 0, len(h.sequence)-1, q=1)
    elif type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
        space[h.name] = hp.choice(h.name, h.choices)
    elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
        space[h.name] = hp.quniform(h.name, h.lower, h.upper, q=1)
    elif type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
        space[h.name] = hp.uniform(h.name, h.lower, h.upper)

y_star_valid, y_star_test = find_nas201_best(nas_bench, dataset)
inc_config = cs.get_default_configuration().get_array().tolist()

# TODO: Cite source
def objective(x):
    config = deepcopy(x)
    for h in cs.get_hyperparameters():
        if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
            config[h.name] = h.sequence[int(x[h.name])]
        elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
            config[h.name] = int(x[h.name])
    y, c, y_test = f(config)
    return {
        'config': config,
        'loss': y,
        'test': y_test,
        'cost': c,
        'status': STATUS_OK}


if args.runs is None:
    trials = Trials()
    best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals=args.n_iters,
                trials=trials)

    res = convert_to_json(trials.results)
    fh = open(os.path.join(output_path, 'run_{}.json'.format(args.run_id)), 'w')
    json.dump(res, fh)
    fh.close()

else:
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if not args.fix_seed:
            np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        trials = Trials()
        best = fmin(objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=args.n_iters,
                    trials=trials)

        res = convert_to_json(trials.results)
        fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
        json.dump(res, fh)
        fh.close()
