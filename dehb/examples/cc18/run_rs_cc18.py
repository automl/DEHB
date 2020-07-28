'''Runs Random Search on XGBoostBenchmark
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

from dehb import DE


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


def randomsearch(dimensions, f, convert_fn, generations, pop_size, budget=None, verbose=True):
    traj = []
    runtime = []
    history = []
    inc_score = np.inf
    inc_config = None
    for i in range(generations * pop_size):
        if verbose:
            print("Iteration #{:<5}/{:<5} - {:<.5f}".format(i+1, generations * pop_size, inc_score))
        config_arr = np.random.uniform(size=dimensions)
        config_obj = convert_fn(config_arr)
        fitness, cost = f(config_obj, budget=budget)
        if fitness < inc_score:
            inc_score = fitness
            inc_config = config_obj
        traj.append(inc_score)
        runtime.append(cost)
        history.append((config_arr.tolist(), float(fitness), float(budget or 0)))

    return np.array(traj), np.array(runtime), np.array(history)


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
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
# pop size and gens included to match DE's number of function evaluation as gens * pop_size
parser.add_argument('--pop_size', default=20, type=int, nargs='?', help='population size')
parser.add_argument('--gens', default=100, type=int, nargs='?',
                    help='(iterations) number of generations for DE to evolve')
parser.add_argument('--max_budget', default=1, type=float,
                    help='the maximum budget for the benchmark')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default='randomsearch', type=str, nargs='?',
                    help='name of folder where files will be dumped')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False
n_estimators = args.n_estimators
max_budget = args.max_budget

task_ids = get_openmlcc18_taskids()
if args.task_id not in task_ids:
    raise "Incorrect task ID. Choose from: {}".format(task_ids)

b = Benchmark(task_id=args.task_id)
# Parameter space to be used by DE
cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

output_path = os.path.join(args.output_path, str(args.task_id), args.folder)
os.makedirs(output_path, exist_ok=True)

# Initializing DE object (for vector_to_configspace access)
de = DE(cs=cs, dimensions=dimensions, f=None, pop_size=None, mutation_factor=None,
        crossover_prob=None, strategy=None)

if args.runs is None:  # for a single run
    if not args.fix_seed:
        np.random.seed(0)
    # Running RS iterations
    traj, runtime, history = randomsearch(dimensions, f, de.vector_to_configspace, args.gens,
                                          args.pop_size, budget=max_budget, verbose=args.verbose)
    fh = open(os.path.join(output_path, 'run_{}.json'.format(args.run_id)), 'w')
    json.dump(calc_test_scores(runtime, history), fh)
    fh.close()
else:  # for multiple runs
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if not args.fix_seed:
            np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        # Running RS iterations
        traj, runtime, history = randomsearch(dimensions, f, de.vector_to_configspace, args.gens,
                                              args.pop_size, budget=max_budget, verbose=args.verbose)
        fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
        json.dump(calc_test_scores(runtime, history), fh)
        fh.close()
        if args.verbose:
            print("Run saved. Resetting...")
        # essential step to not accumulate consecutive runs
        de.reset()

save_configspace(cs, output_path)
