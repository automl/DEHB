'''Runs DE on XGBoostBenchmark
'''

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../HPOBench/'))

import json
import pickle
import argparse
import numpy as np

from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark

from dehb import DE, DEHB


all_task_ids = [126031, 189906, 167155]  # as suggested by Philip


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


# Common objective function for DE representing XGBoostBenchmark
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


def calc_test_scores(runtime, history, de):
    regret_validation = []
    regret_test = []
    inc = np.inf
    test_regret = 1
    for i in range(len(history)):
        config, valid_score, _ = history[i]
        if valid_score < inc:
            inc = valid_score
            config = de.vector_to_configspace(config)
            test_res = None  #b.objective_function_test(config)
            test_score = None  #test_res['function_value']
        regret_test.append(test_score)
        regret_validation.append(inc)
    runtime = np.cumsum(runtime).tolist()
    res = {}
    res['regret_validation'] = regret_validation
    res['regret_test'] = regret_test
    res['runtime'] = runtime
    return res


parser = argparse.ArgumentParser()
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                    nargs='?', help='seed')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--task_id', default=None, type=int,
                    help="specify the OpenML task id to run on from among {}".format(all_task_ids))
parser.add_argument('--n_iters', default=100, type=int, nargs='?',
                    help='number of DEHB iterations')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
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
parser.add_argument('--boundary_fix_type', default='random', type=str, nargs='?',
                    help="strategy to handle solutions outside range {'random', 'clip'}")
parser.add_argument('--eta', default=3, type=int,
                    help='aggressive stopping rate (eta) for Hyperband')
parser.add_argument('--fidelity', default="trees", type=str, choices=["trees", "dataset"],
                    help='Choose the fidelity for XGBoost')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default=None, type=str, nargs='?',
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

# Directory where files will be written
folder = "dehb" if args.folder is None else args.folder

task_ids = all_task_ids if args.task_id is None else [args.task_id]

for task_id in task_ids:
    output_path = os.path.join(args.output_path, str(task_id), folder)
    os.makedirs(output_path, exist_ok=True)

    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        print("Task: {} --- Run {}:\n{}".format(task_id, run_id, "=" * 30))
        if not args.fix_seed:
            np.random.seed(run_id)
        # Initializing benchmark
        rng = np.random.RandomState(seed=run_id)
        b = Benchmark(task_id=task_id, rng=rng)
        # Parameter space to be used by DE
        cs = b.get_configuration_space()
        dimensions = len(cs.get_hyperparameters())
        # Initializing DEHB object
        dehb = DEHB(cs=cs, dimensions=dimensions, f=f, strategy=args.strategy,
                    mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
                    eta=args.eta, min_budget=min_budget, max_budget=max_budget,
                    boundary_fix_type=args.boundary_fix_type)
        # Initializing helper DE object
        de = DE(cs=cs, dimensions=dimensions, f=f, pop_size=10,
                mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
                strategy=args.strategy, budget=max_budget)
        # Starting DEHB optimisation
        traj, runtime, history = dehb.run(iterations=args.n_iters, verbose=args.verbose)
        fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
        json.dump(calc_test_scores(runtime, history, de), fh)
        fh.close()
        print()
