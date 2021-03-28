'''Runs Random Search on XGBoostBenchmark
'''

import os
import sys
import json
import pickle
import argparse
import numpy as np

sys.path.append(os.path.join(os.getcwd(), '../HPOBench/'))
from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark

from dehb import DE


all_task_ids = [189906]  # [126031, 189906, 167155]  # as suggested by Philip


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


def f(config):
    res = b.objective_function(config)
    fitness = res['function_value']
    cost = res['cost']
    return fitness, cost


def calc_test_scores(runtime, history, de, b):
    regret_validation = []
    regret_test = []
    inc = np.inf
    test_regret = 1
    for i in range(len(history)):
        config, valid_score, _ = history[i]
        if valid_score < inc:
            inc = valid_score
            config = de.vector_to_configspace(config)
            test_res = b.objective_function_test(config)
            test_score = test_res['function_value']
        regret_test.append(test_score)
        regret_validation.append(inc)
    runtime = np.cumsum(runtime).tolist()
    res = {}
    res['regret_validation'] = regret_validation
    res['regret_test'] = regret_test
    res['runtime'] = runtime
    return res


def randomsearch(cs, f, iterations, verbose=False):
    traj = []
    runtime = []
    history = []
    for i in range(iterations):
        if verbose:
            print("RS #{}/{}".format(i+1, iterations))
        config = cs.sample_configuration()
        fitness, cost = f(config)
        traj.append(fitness)
        runtime.append(cost)
        history.append((config.get_array().tolist(), float(fitness), float(0)))
    return np.array(traj), np.array(runtime), np.array(history)


parser = argparse.ArgumentParser()
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                    nargs='?', help='seed')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--task_id', default=None, type=int,
                    help="specify the OpenML task id to run on from among {}".format(all_task_ids))
parser.add_argument('--n_iters', default=100, type=int,
                    help="number of Random Search function evaluations")
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default='randomsearch', type=str, nargs='?',
                    help='name of folder where files will be dumped')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False

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
        # Initializing helper DE object
        de = DE(cs=cs, dimensions=dimensions, f=f)
        traj, runtime, history = randomsearch(cs=cs, f=f, iterations=args.n_iters,
                                              verbose=args.verbose)
        fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
        json.dump(calc_test_scores(runtime, history,de, b), fh)
        fh.close()
    print()
