'''Runs BOHB on XGBoostBenchmark
'''

import os
import sys

import json
import pickle
import logging
import argparse
import numpy as np
import ConfigSpace

logging.basicConfig(level=logging.ERROR)

from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker

sys.path.append('dehb/examples/')
from utils import util

sys.path.append(os.path.join(os.getcwd(), '../HPOBench/'))
from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark


all_task_ids = [126031, 189906, 167155]  # as suggested by Philip


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


def convert_to_json(results):
    res = {}
    res['regret_validation'] = np.array(results['losses']).tolist()
    res['regret_test'] = np.array(results['test_losses']).tolist()
    res['runtime'] = np.array(results['cummulative_cost']).tolist()
    return res


def f_trees(config, budget=None):
    global n_estimators, max_budget
    if budget is None:
        budget = max_budget
    fidelity = {"n_estimators": np.round(budget).astype(int), "dataset_fraction": 1}
    res = b.objective_function(config, fidelity)
    fitness = res['function_value']
    cost = res['cost']
    return fitness, cost


def f_dataset(config, budget=None):
    global n_estimators, max_budget
    if budget is None:
        budget = max_budget
    fidelity = {"n_estimators": 128, "dataset_fraction": budget}
    res = b.objective_function(config, fidelity)
    fitness = res['function_value']
    cost = res['cost']
    return fitness, cost


class MyWorker(Worker):
    def __init__(self, f, b, **kwargs):
        super().__init__(**kwargs)
        self.f = f
        self.b = b

    def compute(self, config, budget, **kwargs):
        global max_budget
        if budget is None:
            budget = max_budget
        fitness, cost = self.f(config, budget)
        # res = self.b.objective_function_test(config)
        # print(fitness, cost)
        return ({
            'loss': float(fitness),
            'info': {'cost': float(cost), 'test_loss': None}  #float(res['function_value'])}
        })


parser = argparse.ArgumentParser()
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                    nargs='?', help='seed')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--task_id', default=None, type=int,
                    help="specify the OpenML task id to run on from among {}".format(all_task_ids))
parser.add_argument('--n_iters', default=5, type=int, nargs='?',
                    help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--strategy', default="sampling", type=str, nargs='?',
                    help='optimization strategy for the acquisition function')
parser.add_argument('--min_bandwidth', default=.3, type=float, nargs='?',
                    help='minimum bandwidth for KDE')
parser.add_argument('--bandwidth_factor', default=3, type=int, nargs='?',
                    help='factor multiplied to the bandwidth')
parser.add_argument('--num_samples', default=64, type=int, nargs='?',
                    help='number of samples for the acquisition function')
parser.add_argument('--random_fraction', default=.33, type=float, nargs='?',
                    help='fraction of random configurations')
parser.add_argument('--eta', default=3, type=int,
                    help='aggressive stopping rate (eta) for Hyperband')
parser.add_argument('--fidelity', default="trees", type=str, choices=["trees", "dataset"],
                    help='Choose the fidelity for XGBoost')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default='bohb', type=str, nargs='?',
                    help='name of folder where files will be dumped')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False

if args.fidelity == "trees":
    min_budget = 2
    max_budget = 128
    f = f_trees
else:
    min_budget = 0.1
    max_budget = 1.0
    f = f_dataset

task_ids = all_task_ids if args.task_id is None else [args.task_id]

for task_id in task_ids:
    output_path = os.path.join(args.output_path, str(task_id), args.folder)
    os.makedirs(output_path, exist_ok=True)

    for run_id in range(args.run_start, args.runs + args.run_start):
        print("Task: {} --- Run {}:\n{}".format(task_id, run_id, "=" * 30))
        if not args.fix_seed:
            np.random.seed(run_id)
        # Initializing benchmark
        rng = np.random.RandomState(seed=run_id)
        b = Benchmark(task_id=task_id, rng=rng)
        # Parameter space to be used by DE
        cs = b.get_configuration_space()
        dimensions = len(cs.get_hyperparameters())

        # Starting BOHB stuff
        # hb_run_id = '0'
        hb_run_id = run_id

        NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
        ns_host, ns_port = NS.start()

        print("Creating workers...")
        num_workers = 1
        workers = []
        for i in range(num_workers):
            w = MyWorker(nameserver=ns_host, nameserver_port=ns_port,
                         run_id=hb_run_id, id=i,
                         f=f, b=b)
            w.run(background=True)
            workers.append(w)

        bohb = BOHB(configspace=cs,
                    run_id=hb_run_id,
                    eta=3, min_budget=min_budget, max_budget=max_budget,
                    nameserver=ns_host,
                    nameserver_port=ns_port,
                    # optimization_strategy=args.strategy,
                    num_samples=args.num_samples,
                    random_fraction=args.random_fraction, bandwidth_factor=args.bandwidth_factor,
                    ping_interval=10, min_bandwidth=args.min_bandwidth)
        print("Running BOHB...")
        results = bohb.run(args.n_iters, min_n_workers=num_workers)
        print("Saving file...")
        fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
        json.dump(convert_to_json(util.extract_results_to_pickle(results)), fh)
        fh.close()
        print("Shutting BOHB workers")
        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()
        print()
