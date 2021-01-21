import os
import sys
import json
import pickle
import argparse
import numpy as np

from hpolib.benchmarks.surrogates.paramnet import SurrogateReducedParamNetTime


# Common objective function for DE & DEHB representing Counting Ones benchmark
def f(config, budget=None):
    global max_budget
    d = len(config.get_array())
    if budget is not None:
        fitness = b.objective_function(config, budget=budget)
    else:
        fitness = b.objective_function(config)
        budget = max_budget
    fitness = fitness['function_value']
    cost = budget
    return fitness, cost


def calc_regrets(trajectory):
    valid_scores = []
    inc = np.inf
    for i in range(len(trajectory)):
        score = trajectory[i]
        if score < inc:
            inc = score
        valid_scores.append(inc)
    return valid_scores


def save_json(valid, runtime, output_path, run_id):
    res = {}
    res['validation_score'] = valid
    res['runtime'] = np.cumsum(runtime).tolist()
    fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
    json.dump(res, fh)
    fh.close()


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', help="name of the dataset used",
                    choices=['adult', 'higgs', 'letter', 'mnist', 'optdigits', 'poker'])
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                    nargs='?', help='seed')
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--iter', default=100, type=int, nargs='?',
                    help='number of DEHB iterations')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default='rs', type=str, nargs='?',
                    help='name of folder where files will be dumped')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False

folder = "{}/{}".format(args.dataset, args.folder)
output_path = os.path.join(args.output_path, folder)
os.makedirs(output_path, exist_ok=True)

# Loading benchmark
b = SurrogateReducedParamNetTime(dataset=args.dataset)

# Parameter space to be used by DE
cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

budgets = {  # (min, max)-budget (seconds) for the different data sets
    'adult': (9, 243),
    'higgs': (9, 243),
    'letter': (3, 81),
    'mnist': (9, 243),
    'optdigits': (1, 27),
    'poker': (81, 2187),
}

min_budget, max_budget = budgets[args.dataset]


def run_random_search(iterations, verbose=False):
    global cs, max_budget
    traj = []
    runtime = []
    inc_score = np.inf
    inc_config = None
    for i in range(1, iterations+1):
        config = cs.sample_configuration()
        score, cost = f(config, budget=max_budget)
        if score < inc_score:
            inc_score = score
            inc_config = config
        if verbose:
            print("Iteration {}/{} -> Best found score: {}".format(i, iterations, inc_score))
        traj.append(score)
        runtime.append(cost)
    if verbose:
        print("Best found config: {}".format(inc_config))
        print("Best found score: {}".format(inc_score))
    return traj, runtime


if args.runs is None:  # for a single run
    if not args.fix_seed:
        np.random.seed(args.run_id)
    traj, runtime = run_random_search(iterations=args.iter, verbose=args.verbose)
    valid_scores = calc_regrets(traj)
    save_json(valid_scores, runtime, output_path, args.run_id)

else:  # for multiple runs
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if not args.fix_seed:
            np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        traj, runtime = run_random_search(iterations=args.iter, verbose=args.verbose)
        valid_scores = calc_regrets(traj)
        save_json(valid_scores, runtime, output_path, run_id)

save_configspace(cs, output_path)
