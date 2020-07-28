# Slightly modified version of:
# https://github.com/automl/nas_benchmarks/blob/development/experiment_scripts/run_bohb.py


import os
import sys

import json
import pickle
import logging
import argparse
import numpy as np
import ConfigSpace

logging.basicConfig(level=logging.ERROR)

from hpbandster.optimizers.hyperband import HyperBand
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker

sys.path.append('dehb/examples/')
from utils import util

sys.path.append(os.path.join(os.getcwd(), '../nas201/'))
sys.path.append(os.path.join(os.getcwd(), '../AutoDL-Projects/lib/'))
from nas_201_api import NASBench201API as API
from models import CellStructure, get_search_spaces

from dehb import DE


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
parser.add_argument('--strategy', default="sampling", type=str, nargs='?',
                    help='optimization strategy for the acquisition function')
parser.add_argument('--min_bandwidth', default=.3, type=float, nargs='?',
                    help='minimum bandwidth for KDE')
parser.add_argument('--num_samples', default=64, type=int, nargs='?',
                    help='number of samples for the acquisition function')
parser.add_argument('--random_fraction', default=.33, type=float, nargs='?',
                    help='fraction of random configurations')
parser.add_argument('--min_budget', default=11, type=int, nargs='?',
                    help='minimum budget for NASBench-201')
parser.add_argument('--max_budget', default=199, type=int, nargs='?',
                    help='maximum budget for NASBench-201')
parser.add_argument('--eta', default=3, type=int, nargs='?',
                    help='HB eta')
parser.add_argument('--folder', default='hyperband', type=str, nargs='?',
                    help='name of folder where files will be dumped')
parser.add_argument('--bandwidth_factor', default=3, type=int, nargs='?',
                    help='factor multiplied to the bandwidth')

args = parser.parse_args()
min_budget = args.min_budget
max_budget = args.max_budget
dataset = args.dataset

output_path = os.path.join(args.output_path, args.dataset, args.folder)
os.makedirs(os.path.join(output_path), exist_ok=True)

# Loading NAS-201
api = API(args.data_dir)
search_space = get_search_spaces('cell', 'nas-bench-201')

# Parameter space to be used by DE
cs = get_configuration_space(args.max_nodes, search_space)
dimensions = len(cs.get_hyperparameters())
config2structure = config2structure_func(args.max_nodes)

y_star_valid, y_star_test = find_nas201_best(api, dataset)
inc_config = cs.get_default_configuration().get_array().tolist()

class MyWorker(Worker):
    def compute(self, config, budget, **kwargs):
        global dataset, api, max_budget
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

        info = api.get_more_info(arch_index, dataset, iepoch=max_budget,
                                 use_12epochs_result=False, is_random=False)
        test_score = 1 - info['test-accuracy'] / 100
        return ({
            'loss': float(fitness),
            'info': {'cost': float(cost), 'test_loss': float(test_score)}
        })


runs = args.runs
for run_id in range(runs):
    print("Run {:>3}/{:>3}".format(run_id+1, runs))

    # hb_run_id = '0'
    hb_run_id = run_id

    NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
    ns_host, ns_port = NS.start()

    num_workers = 1

    workers = []
    for i in range(num_workers):
        w = MyWorker(nameserver=ns_host, nameserver_port=ns_port,
                     run_id=hb_run_id,
                     id=i)
        w.run(background=True)
        workers.append(w)

    HB = HyperBand(configspace=cs,
                run_id=hb_run_id,
                eta=3, min_budget=min_budget, max_budget=max_budget,
                nameserver=ns_host,
                nameserver_port=ns_port,
                ping_interval=10)

    results = HB.run(args.n_iters, min_n_workers=num_workers)

    HB.shutdown(shutdown_workers=True)
    NS.shutdown()

    # fh = open(os.path.join(output_path, 'hyperband_run_%d.pkl' % run_id), 'wb')
    # pickle.dump(util.extract_results_to_pickle(results), fh)
    fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
    json.dump(convert_to_json(util.extract_results_to_pickle(results)), fh)
    fh.close()
    print("Run saved. Resetting...")
