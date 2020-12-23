'''Runs DEHB on BNN
'''

import os
import sys
import json
import pickle
import argparse
import numpy as np
import ConfigSpace

from hpolib.benchmarks.ml.bnn_benchmark import BNNOnBostonHousing, BNNOnProteinStructure
from hpolib.benchmarks.ml.bnn_benchmark import BNNOnToyFunction, BNNOnYearPrediction

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO as SMAC


def objective_function(config):
    global max_budget, b
    fitness = b.objective_function(config, budget=int(max_budget))['function_value']
    cost = max_budget
    return float(fitness), {
        "runtime": cost,
        "config": config.get_dictionary()
    }


def calc_regrets(data):
    global b, cs
    valid_scores = []
    test_scores = []
    inc = np.inf
    test_regret = np.inf
    runtimes = []
    for k, v in data.items():
        valid_regret = v.cost
        if valid_regret < inc:
            inc = valid_regret
            config = ConfigSpace.Configuration(cs, v.additional_info["config"])
            res = b.objective_function_test(config)
            test_regret = res["function_value"]
        test_scores.append(test_regret)
        valid_scores.append(inc)
        runtimes.append(v.additional_info["runtime"])
    return valid_scores, test_scores, runtimes


def save_json(valid, test, runtime, output_path, run_id):
    res = {}
    res['regret_validation'] = valid
    res['regret_test'] = test
    res['runtime'] = np.cumsum(runtime).tolist()
    fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
    json.dump(res, fh)
    fh.close()


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(output_path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help="name of the dataset used", default='bostonhousing',
                    choices=['toyfunction', 'bostonhousing', 'proteinstructure', 'yearprediction'])
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--n_iters', default=100, type=int, nargs='?',
                    help='number of DEHB iterations')
parser.add_argument('--min_budget', default=500, type=int, nargs='?',
                    help='minimum budget')
parser.add_argument('--max_budget', default=10000, type=int, nargs='?',
                    help='maximum budget')
parser.add_argument('--verbose', action="store_true",
                    help='to print progress or not')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='name of directory where results will be directed')
parser.add_argument('--folder', default="smac", type=str, nargs='?',
                    help='name of folder where files will be dumped')

args = parser.parse_args()
min_budget = args.min_budget
max_budget = args.max_budget
dataset = args.dataset

# Directory where files will be written
folder = args.folder
output_path = os.path.join(args.output_path, args.dataset, folder)
os.makedirs(output_path, exist_ok=True)

# Parameter space to be used by DE
if args.dataset == 'bostonhousing':
    b = BNNOnBostonHousing()
elif args.dataset == 'proteinstructure':
    b = BNNOnProteinStructure()
elif args.dataset == 'toyfunction':
    b = BNNOnToyFunction()
else:
    b = BNNOnYearPrediction()

cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

scenario = Scenario({"run_obj": "quality",
                     "runcount-limit": args.n_iters,
                     "cs": cs,
                     "deterministic": "false",
                     "initial_incumbent": "RANDOM",
                     "output_dir": ""})


if args.runs is None:  # for a single run
    np.random.seed(args.run_id)
    smac = SMAC(scenario=scenario, tae_runner=objective_function)
    smac.optimize()
    valid_scores, test_scores, runtimes = calc_regrets(smac.runhistory.data)
    save_json(valid_scores, test_scores, runtimes, output_path, args.run_id)

else:  # for multiple runs
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        smac = SMAC(scenario=scenario, tae_runner=objective_function)
        smac.optimize()
        valid_scores, test_scores, runtimes = calc_regrets(smac.runhistory.data)
        save_json(valid_scores, test_scores, runtimes, output_path, run_id)

        if args.verbose:
            print("Run saved. Resetting...")

save_configspace(cs, output_path)
