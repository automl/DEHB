'''Runs TPE on XGBoostBenchmark
'''

import os
import sys

import json
import pickle
import logging
import argparse
import numpy as np
import ConfigSpace
from copy import deepcopy

logging.basicConfig(level=logging.ERROR)

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

sys.path.append('dehb/examples/')
from utils import util

sys.path.append(os.path.join(os.getcwd(), '../HPOlib3/'))
from hpolib.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark
from hpolib.util.openml_data_manager import get_openmlcc18_taskids


# task_ids = get_openmlcc18_taskids()
task_ids = [126031, 189906, 167155]  # as suggested by Philip


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


def objective(x):
    global b, de, max_budget
    config = deepcopy(x)
    for h in cs.get_hyperparameters():
        if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
            config[h.name] = h.sequence[int(x[h.name])]
        elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
            config[h.name] = int(x[h.name])
    print(config)
    temp = cs.sample_configuration()
    temp._values = config
    config = temp
    res = b.objective_function(config, n_estimators=n_estimators, subsample=max_budget)
    fitness = res['function_value']
    cost = res['cost']
    y_test = b.objective_function_test(config, n_estimators=n_estimators)['function_value']
    return {
        'config': config,
        'loss': fitness,
        'test': y_test,
        'cost': cost,
        'status': STATUS_OK}


def convert_to_json(results):
    global y_star_valid, y_star_test
    regret_validation = []
    regret_test = []
    runtime = []
    inc = np.inf
    test_regret = 1
    validation_regret = 1

    for i in range(len(results)):
        validation_regret = results[i]['loss']
        if validation_regret <= inc:
            inc = validation_regret
            test_regret = results[i]['test']
        regret_validation.append(inc)
        regret_test.append(test_regret)
        runtime.append(results[i]['cost'])

    res = {}
    res['regret_validation'] = regret_validation
    res['regret_test'] = regret_test
    res['runtime'] = np.cumsum(runtime).tolist()
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
parser.add_argument('--n_iters', default=5, type=int, nargs='?',
                    help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--max_budget', default=1, type=float,
                    help='the maximum budget for the benchmark')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default='tpe', type=str, nargs='?',
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

space = {}
for h in cs.get_hyperparameters():
    if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
        space[h.name] = hp.quniform(h.name, 0, len(h.sequence)-1, q=1)
    elif type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
        space[h.name] = hp.choice(h.name, h.choices)
    elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
        space[h.name] = hp.quniform(h.name, h.lower, h.upper, q=1)
    elif type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
        space[h.name] = hp.uniform(h.name, h.lower, h.upper)

output_path = os.path.join(args.output_path, str(args.task_id), args.folder)
os.makedirs(output_path, exist_ok=True)


if args.runs is None:
    if not args.fix_seed:
        np.random.seed(args.run_id)
    trials = Trials()
    best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals=args.n_iters,
                trials=trials)

    res = convert_to_json(trials.results)
    fh = open(os.path.join(output_path, 'run_{}.json'.format(args.run_id)), 'w')
    json.dump(res, fh)
    fh.close()

else:
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if not args.fix_seed:
            np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        trials = Trials()
        best = fmin(objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=args.n_iters,
                    trials=trials)

        res = convert_to_json(trials.results)
        fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
        json.dump(res, fh)
        fh.close()
