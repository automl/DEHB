import os
import sys
import json
import pickle
import argparse
import numpy as np

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO as SMAC
from smac.initial_design.latin_hypercube_design import LHDesign

from hpolib.benchmarks.synthetic_functions.counting_ones import CountingOnes


def objective_function(config, **kwargs):
    global b, max_budget
    fitness = b.objective_function(config, budget=max_budget)['function_value']
    test_fitness = b.objective_function_test(config)['function_value']
    return float(fitness), {
        "runtime": float(max_budget),
        "test_score": float(test_fitness)
    }


def calc_regrets(data):
    global dimensions
    global_best = -dimensions
    valid_regret = []
    test_regret = []
    runtimes = []
    val_inc = np.inf
    test_inc = np.inf
    for k, v in data.items():
        if v.cost < val_inc:
            val_inc = v.cost
        if v.additional_info["test_score"] < test_inc:
            test_inc = v.additional_info["test_score"]
        valid_regret.append(val_inc - global_best)
        test_regret.append(test_inc - global_best)
        runtimes.append(v.additional_info["runtime"])
    return valid_regret, test_regret, runtimes


def save_json(valid, test, runtime, output_path, run_id):
    res = dict()
    res['regret_validation'] = valid
    res['regret_test'] = test
    res['runtime'] = np.cumsum(runtime).tolist()
    fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
    json.dump(res, fh)
    fh.close()


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


parser = argparse.ArgumentParser()
parser.add_argument('--n_cat', type=int, default=4,
                    help='Number of categorical parameters in the search space.')
parser.add_argument('--n_cont', type=int, default=4,
                    help='Number of continuous parameters in the search space.')
parser.add_argument('--n_iters', type=int, default=4,
                    help='number of iterations performed.')
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--run_start', default=0, type=int, nargs='?', help='starting index for run id')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--verbose', action='store_true',
                    help='to print progress or not')
parser.add_argument('--folder', default="smac", type=str, nargs='?',
                    help='name of folder where files will be dumped')

args = parser.parse_args()

dim_folder = "{}+{}".format(args.n_cont, args.n_cat)
folder = "{}/{}".format(dim_folder, args.folder)
output_path = os.path.join(args.output_path, folder)
os.makedirs(output_path, exist_ok=True)

# Loading benchmark
b = CountingOnes()
cs = b.get_configuration_space(n_continuous=args.n_cont, n_categorical=args.n_cat)
dimensions = len(cs.get_hyperparameters())

min_budget = 576 / dimensions
max_budget = 93312 / dimensions

scenario = Scenario({"run_obj": "quality",
                     "runcount-limit": args.n_iters,
                     "cs": cs,
                     "deterministic": "false",
                     "initial_incumbent": "RANDOM",
                     "output_dir": ""})

if args.runs is None:  # for a single run
    np.random.seed(args.run_id)
    if dimensions < 40:
        smac = SMAC(scenario=scenario, tae_runner=objective_function)
    else:
        smac = SMAC(scenario=scenario, tae_runner=objective_function, initial_design=LHDesign)
    smac.optimize()
    valid_scores, test_scores, runtimes = calc_regrets(smac.runhistory.data)
    save_json(valid_scores, test_scores, runtimes, output_path, args.run_id)

else:  # for multiple runs
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        if dimensions < 40:
            smac = SMAC(scenario=scenario, tae_runner=objective_function)
        else:
            smac = SMAC(scenario=scenario, tae_runner=objective_function, initial_design=LHDesign)
        smac.optimize()
        valid_scores, test_scores, runtimes = calc_regrets(smac.runhistory.data)
        save_json(valid_scores, test_scores, runtimes, output_path, run_id)
        if args.verbose:
            print("Run saved. Resetting...")

save_configspace(cs, output_path)
