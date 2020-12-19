
import os
import sys

import json
import pickle
import logging
import argparse
import numpy as np
import ConfigSpace

logging.basicConfig(level=logging.ERROR)

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncDict

sys.path.append('dehb/examples/')
from utils import util

sys.path.append(os.path.join(os.getcwd(), '../HpBandSter/icml_2018_experiments/experiments/workers'))
from base_worker import BaseWorker

sys.path.append(os.path.join(os.getcwd(), '../nas201/'))
sys.path.append(os.path.join(os.getcwd(), '../AutoDL-Projects/lib/'))
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


def convert_to_json(results):
    global y_star_valid, y_star_test
    res = {}
    res['regret_validation'] = np.array(results['losses'] - y_star_valid).tolist()
    res['regret_test'] = np.array(results['test_losses'] - y_star_test).tolist()
    res['runtime'] = np.array(results['cummulative_cost']).tolist()
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

parser.add_argument('--start', default=0, type=int, nargs='?',
                    help='unique number to start run_id')
parser.add_argument('--runs', default=1, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--dataset', default='cifar10-valid', type=str, nargs='?',
                    choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'],
                    help='choose the dataset')
parser.add_argument('--max_nodes', default=4, type=int, nargs='?',
                    help='maximum number of nodes in the cell')
parser.add_argument('--n_iters', default=5, type=int, nargs='?',
                    help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', type=str, nargs='?',
                    default="../nas201/NAS-Bench-201-v1_1-096897.pth",
                    help='specifies the path to the tabular data')
parser.add_argument('--n_trees', default=10, type=int, nargs='?',
                    help='number of trees for the random forest')
parser.add_argument('--random_fraction', default=.33, type=float, nargs='?',
                    help='fraction of random configurations')
parser.add_argument('--max_feval', default=4, type=int, nargs='?',
                    help='maximum number of function evaluation per configuration')
parser.add_argument('--folder', default='smac', type=str, nargs='?',
                    help='name of folder where files will be dumped')

args = parser.parse_args()
dataset = args.dataset

output_path = os.path.join(args.output_path, args.dataset, args.folder)
os.makedirs(os.path.join(output_path), exist_ok=True)
args.working_directory = output_path
args.method = "smac"
args.min_budget = 11
args.max_budget = 199
args.eta = 3
args.num_iterations = args.n_iters

# Loading NAS-201
api = API(args.data_dir)
search_space = get_search_spaces('cell', 'nas-bench-201')

# Parameter space to be used by DE
cs = get_configuration_space(args.max_nodes, search_space)
dimensions = len(cs.get_hyperparameters())
config2structure = config2structure_func(args.max_nodes)

y_star_valid, y_star_test = find_nas201_best(api, dataset)
inc_config = cs.get_default_configuration().get_array().tolist()

scenario = Scenario({"run_obj": "quality",
                     "runcount-limit": args.n_iters,
                     "cs": cs,
                     "deterministic": "false",
                     "initial_incumbent": "RANDOM",
                     "output_dir": ""})

class NAS201Worker(BaseWorker):
    def __init__(self, api, cs, **kwargs):
        super().__init__(benchmark=api, configspace=cs, **kwargs)
        self.api = api
        self.cs = cs

    def compute(self, config, **kwargs):
        # global dataset, api
        structure = config2structure(config)
        arch_index = self.api.query_index_by_arch(structure)
        # From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/R_EA.py
        ## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
        info = self.api.get_more_info(arch_index, dataset, #iepoch=budget,
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

        # info = api.get_more_info(arch_index, dataset, #iepoch=max_budget,
        #                          use_12epochs_result=False, is_random=False)
        test_score = 1 - info['test-accuracy'] / 100

        return ({
            'loss': float(fitness),
            'info': {'cost': float(cost), 'test_loss': float(test_score)}
        })

start = args.start
runs = args.runs
for run_id in range(start, runs):
    print("Run {:>3}/{:>3}".format(run_id+1, runs))

    worker = NAS201Worker(api=api, cs=cs, measure_test_loss=False, run_id=run_id, max_budget=199)
    args.run_id = run_id
    result = util.run_experiment(args, worker, output_path, smac_deterministic=False)
    # tae = ExecuteTAFuncDict(objective_function, use_pynisher=False)
    # smac = SMAC(scenario=scenario, tae_runner=tae)
    #
    # # probability for random configurations
    # smac.solver.random_configuration_chooser.prob = args.random_fraction
    # smac.solver.model.rf_opts.num_trees = args.n_trees
    # # only 1 configuration per SMBO iteration
    # smac.solver.scenario.intensification_percentage = 1e-10
    # smac.solver.intensifier.min_chall = 1
    # # maximum number of function evaluations per configuration
    # smac.solver.intensifier.maxR = args.max_feval
    #
    # smac.optimize()

    # fh = open(os.path.join(output_path, 'run_%d.json' % run_id), 'w')
    # json.dump(res, fh)
    # fh.close()

    with open(os.path.join(output_path, 'smac_run_{}.pkl'.format(run_id)), "rb") as f:
        result = pickle.load(f)
    fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
    json.dump(convert_to_json(result), fh)
    fh.close()
    os.remove(os.path.join(output_path, 'smac_run_{}.pkl'.format(run_id)))

    print("Run saved. Resetting...")
