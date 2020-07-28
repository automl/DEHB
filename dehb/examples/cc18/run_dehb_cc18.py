'''Runs DE on XGBoostBenchmark
'''

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../HPOlib3/'))

import json
import pickle
import argparse
import numpy as np

from hpolib.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark
from hpolib.util.openml_data_manager import get_openmlcc18_taskids

from dehb import DE, DEHB, DEHB_0, DEHB_1, DEHB_2, DEHB_3


# task_ids = get_openmlcc18_taskids()
task_ids = [126031, 189906, 167155]  # as suggested by Philip


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


# Common objective function for DE representing XGBoostBenchmark
def f(config, budget=None):
    global n_estimators, max_budget
    if budget is None:
        budget = max_budget
    res = b.objective_function(config, n_estimators=n_estimators, subsample=budget)
    fitness = res['function_value']
    cost = res['cost']
    return fitness, cost


def calc_test_scores(runtime, history):
    global n_estimators
    regret_validation = []
    regret_test = []
    inc = np.inf
    test_regret = 1
    for i in range(len(history)):
        config, valid_score, _ = history[i]
        if valid_score < inc:
            inc = valid_score
            config = de.vector_to_configspace(config)
            test_res = b.objective_function_test(config, n_estimators=n_estimators)
            test_score = test_res['function_value']
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
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--task_id', default=task_ids[0], type=int,
                    help="specify the OpenML task id to run on from among {}".format(task_ids))
parser.add_argument('--n_estimators', default=64, type=int,
                    help="specify the number of estimators XGBoost will be trained with")
parser.add_argument('--iter', default=100, type=int, nargs='?',
                    help='number of DEHB iterations')
parser.add_argument('--gens', default=100, type=int, nargs='?',
                    help='(iterations) number of generations for DE to evolve')
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
parser.add_argument('--eta', default=3, type=int,
                    help='aggressive stopping rate (eta) for Hyperband')
parser.add_argument('--min_budget', default=0.1, type=float,
                    help='the minimum budget for the benchmark')
parser.add_argument('--max_budget', default=1, type=float,
                    help='the maximum budget for the benchmark')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default=None, type=str, nargs='?',
                    help='name of folder where files will be dumped')
parser.add_argument('--version', default=None, type=str, nargs='?',
                    help='DEHB version to run')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False
n_estimators = args.n_estimators
min_budget = args.min_budget
max_budget = args.max_budget

task_ids = get_openmlcc18_taskids()
if args.task_id not in task_ids:
    raise "Incorrect task ID. Choose from: {}".format(task_ids)

b = Benchmark(task_id=args.task_id)
# Parameter space to be used by DE
cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

# Directory where files will be written
if args.folder is None:
    folder = "dehb"
    if args.version is not None:
        folder = "dehb_v{}".format(args.version)
else:
    folder = args.folder

output_path = os.path.join(args.output_path, str(args.task_id), folder)
os.makedirs(output_path, exist_ok=True)

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
        np.random.seed(args.run_id)
    # Running DE iterations
    traj, runtime, history = dehb.run(iterations=args.iter, verbose=args.verbose)
    fh = open(os.path.join(output_path, 'run_{}.json'.format(args.run_id)), 'w')
    json.dump(calc_test_scores(runtime, history), fh)
    fh.close()
else:  # for multiple runs
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if not args.fix_seed:
            np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        # Running DE iterations
        traj, runtime, history = dehb.run(iterations=args.iter, verbose=args.verbose)
        fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
        json.dump(calc_test_scores(runtime, history), fh)
        fh.close()
        if args.verbose:
            print("Run saved. Resetting...")
        # essential step to not accumulate consecutive runs
        de.reset()

save_configspace(cs, output_path)
