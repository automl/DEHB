import os
import sys
import json
import pickle
import argparse
import numpy as np

import ConfigSpace

from hpolib.benchmarks.surrogates.svm import SurrogateSVM as surrogate

from dehb import DE
from dehb import DEHB, DEHB_0, DEHB_1


# Common objective function for DE & DEHB representing SVM Surrogates benchmark
def f(config, budget=None):
    if budget is not None:
        res = b.objective_function(config, dataset_fraction=budget)
    else:
        res = b.objective_function(config)
    fitness, cost = res['function_value'], res['cost']
    return fitness, cost


def calc_test_scores(history):
    global de, b
    valid_scores = []
    test_scores = []
    test_error = 1
    inc = np.inf
    for i in range(len(history)):
        valid_error = history[i][1]
        if valid_error < inc:
            inc = valid_error
            config = de.vector_to_configspace(history[i][0])
            res = b.objective_function_test(config)
            test_error = res['function_value']
        test_scores.append(test_error)
    return test_scores


def save_json(valid, test, runtime, output_path, run_id):
    res = {}
    res['validation_score'] = valid.tolist()
    res['test_score'] = test
    res['runtime'] = np.cumsum(runtime).tolist()
    fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
    json.dump(res, fh)
    fh.close()


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


parser = argparse.ArgumentParser()
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                    nargs='?', help='seed')
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--iter', default=20, type=int, nargs='?',
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
parser.add_argument('--gens', default=1, type=int, nargs='?',
                    help='DE generations in each DEHB iteration')
parser.add_argument('--eta', default=3, type=int, nargs='?',
                    help='SH parameter')
parser.add_argument('--min_budget', default=1/512, type=float, nargs='?',
                    help='DEHB max budget')
parser.add_argument('--max_budget', default=1, type=float, nargs='?',
                    help='DEHB max budget')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default=None, type=str, nargs='?',
                    help='name of folder where files will be dumped')
parser.add_argument('--version', default=None, type=str, nargs='?',
                    help='version of DEHB to run')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False

if args.folder is None:
    if args.version is not None:
        folder = "dehb"
    else:
        folder = "dehb_v{}".format(args.version)
else:
    folder = args.folder

dehbs = {None: DEHB, "0": DEHB_0, "1": DEHB_1}
DEHB = dehbs[args.version]

output_path = os.path.join(args.output_path, folder)
os.makedirs(output_path, exist_ok=True)

# Loading benchmark
b = surrogate()

# Parameter space to be used by DE
cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())


# Initializing DEHB object
dehb = DEHB(cs=cs, dimensions=dimensions, f=f, strategy=args.strategy,
            mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
            eta=args.eta, min_budget=args.min_budget, max_budget=args.max_budget,
            generations=args.gens)

# Helper DE object for vector to config mapping
de = DE(cs=cs, b=b, f=f)


if args.runs is None:  # for a single run
    if not args.fix_seed:
        np.random.seed(0)
    # Running DE iterations
    traj, runtime, history = dehb.run(iterations=args.iter, verbose=args.verbose)
    test_scores = calc_test_scores(history)

    save_json(traj, test_scores, runtime, output_path, args.run_id)

else:  # for multiple runs
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if not args.fix_seed:
            np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        # Running DE iterations
        traj, runtime, history = dehb.run(iterations=args.iter, verbose=args.verbose)
        test_scores = calc_test_scores(history)

        save_json(traj, test_scores, runtime, output_path, run_id)

        if args.verbose:
            print("Run saved. Resetting...")
        # essential step to not accumulate consecutive runs

        dehb.reset()

save_configspace(cs, output_path)
