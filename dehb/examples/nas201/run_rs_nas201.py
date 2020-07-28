#################
# Random Search #
#################
# This script is modified from a script authored by Xuanyi Dong
# https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/R_EA.py

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

from nas_201_api import NASBench201API as API
from models import CellStructure, get_search_spaces


class Model(object):

  def __init__(self):
    self.arch = None
    self.accuracy = None

  def __str__(self):
    """Prints a readable version of this bitstring."""
    return '{:}'.format(self.arch)



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

# This function is to mimic the training and evaluatinig procedure for a single architecture `arch`.
# The time_cost is calculated as the total training time for a few (e.g., 12 epochs) plus the evaluation time for one epoch.
# For use_converged_LR = True, the architecture is trained for 12 epochs, with LR being decaded from 0.1 to 0.
#       In this case, the LR schedular is converged.
# For use_converged_LR = False, the architecture is planed to be trained for 200 epochs, but we early stop its procedure.
#
def train_and_eval(arch, nas_bench, dataname='cifar10-valid', use_converged_LR=False):
  global max_budget
  if nas_bench is not None:
    arch_index, nepoch = nas_bench.query_index_by_arch( arch ), 199
    assert arch_index >= 0, 'can not find this arch : {:}'.format(arch)
    info = nas_bench.get_more_info(arch_index, dataname, iepoch=max_budget,
                                   use_12epochs_result=False, is_random=True)
    try:
      valid_acc = info['valid-accuracy']
    except:
      valid_acc = info['valtest-accuracy']

    time_cost = info['train-all-time']
    try:
      time_cost += info['valid-all-time']
    except:
      time_cost += info['valtest-all-time']
  else:
    # train a model from scratch.
    raise ValueError('NOT IMPLEMENT YET')
  return valid_acc, time_cost


def random_architecture_func(max_nodes, op_names):
  # return a random architecture
  def random_architecture():
    genotypes = []
    for i in range(1, max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        op_name  = random.choice( op_names )
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return CellStructure( genotypes )
  return random_architecture


def random_search(time_budget, nas_bench, dataname):
  """Algorithm for regularized evolution (i.e. aging evolution).

  Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
  Classifier Architecture Search".

  Args:
    cycles: the number of cycles the algorithm should run for.
    population_size: the number of individuals to keep in the population.
    sample_size: the number of individuals that should participate in each tournament.
    time_budget: the upper bound of searching cost

  Returns:
    history: a list of `Model` instances, representing all the models computed
        during the evolution experiment.
  """
  population = collections.deque()
  history, total_time_cost = [], 0  # Not used by the algorithm, only used to report results.

  # Initialize the population with random models.
  while total_time_cost < time_budget:
    model = Model()
    model.arch = random_arch()
    model.accuracy, time_cost = train_and_eval(model.arch, nas_bench, dataname)
    population.append(model)
    history.append((model, time_cost))
    total_time_cost += time_cost

  return history, total_time_cost

def convert_to_json(history):
    global y_star_valid, y_star_test, max_budget
    regret_validation = []
    regret_test = []
    runtime = []
    inc = np.inf
    test_regret = 1
    validation_regret = 1

    for i in range(len(history)):
        architecture, cost = history[i]
        validation_regret = (1 - (architecture.accuracy / 100)) - y_star_valid
        # validation_regret = architecture.accuracy - y_star_valid
        if validation_regret <= inc:
            inc = validation_regret
            arch_index = nas_bench.query_index_by_arch(architecture.arch)
            info = nas_bench.get_more_info(arch_index, dataset, max_budget, False, False)
            test_regret = (1 - (info['test-accuracy'] / 100)) - y_star_test
        regret_validation.append(inc)
        regret_test.append(test_regret)
        runtime.append(cost)

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
                    help='maximum number of nodes in the cell for NASBench-201')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--output_path', default="./results/", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../nas201/NAS-Bench-201-v1_1-096897.pth",
                    type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--time_budget', default=1e7, type=float, nargs='?', help='time budget')
parser.add_argument('--min_budget', default=11, type=int, nargs='?',
                    help='minimum budget for NASBench-201')
parser.add_argument('--max_budget', default=199, type=int, nargs='?',
                    help='maximum budget for NASBench-201')
parser.add_argument('--folder', default='randomsearch', type=str, nargs='?',
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
random_arch = random_architecture_func(args.max_nodes, search_space)

# Getting configuration space
cs = get_configuration_space(args.max_nodes, search_space)
dimensions = len(cs.get_hyperparameters())
config2structure = config2structure_func(args.max_nodes)

y_star_valid, y_star_test = find_nas201_best(nas_bench, dataset)
inc_config = cs.get_default_configuration().get_array().tolist()


if args.runs is None:
    history, total_cost = random_search(args.time_budget, nas_bench, dataset)

    res = convert_to_json(history)
    fh = open(os.path.join(output_path, 'run_{}.json'.format(args.run_id)), 'w')
    json.dump(res, fh)
    fh.close()

else:
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if not args.fix_seed:
            np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        history, total_cost = random_search(args.time_budget, nas_bench, dataset)

        res = convert_to_json(history)
        fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
        json.dump(res, fh)
        fh.close()
