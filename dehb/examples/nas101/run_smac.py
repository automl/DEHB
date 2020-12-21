# Slightly modified version of:
# https://github.com/automl/nas_benchmarks/blob/development/experiment_scripts/run_smac.py

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks/'))
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks-development/'))

import json
import pickle
import argparse
import numpy as np

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO as SMAC

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C

sys.path.append('dehb/examples/')
from utils import util

sys.path.append(os.path.join(os.getcwd(), '../HpBandSter/icml_2018_experiments/experiments/workers'))
from base_worker import BaseWorker


def convert_to_json(results, y_star_valid, y_star_test):
    res = {}
    res['regret_validation'] = np.array(np.array(results['losses']) - y_star_valid).tolist()
    res['regret_test'] = np.array(np.array(results['test_losses']) - y_star_test).tolist()
    res['runtime'] = np.array(results['cummulative_cost']).tolist()
    return res


class NAS101Worker(BaseWorker):
    def __init__(self, benchmark, cs, **kwargs):
        super().__init__(benchmark=benchmark, configspace=cs, **kwargs)
        self.b = benchmark

    def compute(self, config, **kwargs):
        valid_score, cost = self.b.objective_function(config)
        test_score, _ = self.b.objective_function_test(config)
        return ({
            'loss': float(valid_score),
            'info': {'cost': float(cost), 'test_loss': float(test_score)}
        })


parser = argparse.ArgumentParser()

parser.add_argument('--start', default=0, type=int, nargs='?',
                    help='unique number to start the run_id')
parser.add_argument('--runs', default=1, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--benchmark', default="nas_cifar10a", type=str, nargs='?',
                    help='specifies the benchmark')
parser.add_argument('--n_iters', default=5, type=int, nargs='?',
                    help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../nas_benchmarks-development/"
                                          "tabular_benchmarks/fcnet_tabular_benchmarks/",
                    type=str, nargs='?', help='specifies the path to the tabular data')
args = parser.parse_args()

if args.benchmark == "nas_cifar10a":
    min_budget = 4
    max_budget = 108
    b = NASCifar10A(data_dir=args.data_dir, multi_fidelity=False)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test

elif args.benchmark == "nas_cifar10b":
    min_budget = 4
    max_budget = 108
    b = NASCifar10B(data_dir=args.data_dir, multi_fidelity=False)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test

elif args.benchmark == "nas_cifar10c":
    min_budget = 4
    max_budget = 108
    b = NASCifar10C(data_dir=args.data_dir, multi_fidelity=False)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test

elif args.benchmark == "protein_structure":
    min_budget = 3
    max_budget = 100
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)
    _, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "slice_localization":
    min_budget = 3
    max_budget = 100
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)
    _, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "naval_propulsion":
    min_budget = 3
    max_budget = 100
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)
    _, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "parkinsons_telemonitoring":
    min_budget = 3
    max_budget = 100
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)
    _, y_star_valid, y_star_test = b.get_best_configuration()

output_path = os.path.join(args.output_path, "smac")
os.makedirs(os.path.join(output_path), exist_ok=True)

cs = b.get_configuration_space()

scenario = Scenario({"run_obj": "quality",
                     "runcount-limit": args.n_iters,
                     "cs": cs,
                     "deterministic": "false",
                     "initial_incumbent": "RANDOM",
                     "output_dir": ""})

start = args.start
runs = args.runs
assert runs > start
for run_id in range(start, runs):
    print("Run {:>3}/{:>3}".format(run_id+1, runs))

    worker = NAS101Worker(
        benchmark=b, cs=cs, measure_test_loss=False, run_id=run_id, max_budget=max_budget
    )
    args.run_id = run_id
    args.min_budget = min_budget
    args.max_budget = max_budget
    args.eta = 3
    result = util.run_experiment(args, worker, output_path, smac_deterministic=False)

    with open(os.path.join(output_path, 'smac_run_{}.pkl'.format(run_id)), "rb") as f:
        result = pickle.load(f)
    fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
    json.dump(convert_to_json(result, y_star_valid, y_star_test), fh)
    fh.close()
    os.remove(os.path.join(output_path, 'smac_run_{}.pkl'.format(run_id)))

    print("Run saved. Resetting...")
    b.reset_tracker()
