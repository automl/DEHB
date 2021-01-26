import os
import sys
import json
import pickle
import argparse
import numpy as np

from hpolib.benchmarks.synthetic_functions.counting_ones import CountingOnes


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


def f_test(config, budget=None):
    global max_budget
    d = len(config.get_array())
    if budget is not None:
        fitness = b.objective_function_test(config, budget=budget)
    else:
        fitness = b.objective_function_test(config)
        budget = max_budget
    fitness = fitness['function_value']
    cost = budget
    return fitness, cost


def calc_regrets(trajectory, test_trajectory):
    global cs
    d = len(cs.get_hyperparameters())
    valid_scores = []
    test_scores = []
    inc = np.inf
    test_inc = np.inf
    for i in range(len(trajectory)):
        valid_regret = (trajectory[i] + d) / d
        test_regret = (test_trajectory[i] + d) / d
        if valid_regret < inc:
            inc = valid_regret
            test_inc = test_regret
        valid_scores.append(inc)
        test_scores.append(test_inc)
    return valid_scores, test_scores


def save_json(valid, test, runtime, output_path, run_id):
    res = {}
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
parser.add_argument('--n_cont', default=4, type=int, nargs='?',
                    help='number of continuous variables')
parser.add_argument('--n_cat', default=4, type=int, nargs='?',
                    help='number of categorical variables')
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

dim_folder = "{}+{}".format(args.n_cont, args.n_cat)

folder = "{}/{}".format(dim_folder, args.folder)
output_path = os.path.join(args.output_path, folder)
os.makedirs(output_path, exist_ok=True)

# Loading benchmark
b = CountingOnes()

# Parameter space to be used by DE
cs = b.get_configuration_space(n_continuous=args.n_cont, n_categorical=args.n_cat)
dimensions = len(cs.get_hyperparameters())

min_budget = 576 / dimensions
max_budget = 93312 / dimensions

y_star_test = -dimensions  # incorporated in regret_calc as normalized regret: (f(x) + d) / d


def run_random_search(iterations, verbose=False):
    global cs, max_budget
    traj = []
    test_traj = []
    runtime = []
    inc_score = np.inf
    inc_config = None
    for i in range(1, iterations+1):
        config = cs.sample_configuration()
        score, cost = f(config, budget=max_budget)
        test_score, _ = f_test(config)
        if score < inc_score:
            inc_score = score
            inc_config = config
        if verbose:
            print("Iteration {}/{} -> Best found score: {}".format(i, iterations, inc_score))
        traj.append(score)
        test_traj.append(test_score)
        runtime.append(cost)
    if verbose:
        print("Best found config: {}".format(inc_config))
        print("Best found score: {}".format(inc_score))
    return traj, test_traj, runtime


if args.runs is None:  # for a single run
    if not args.fix_seed:
        np.random.seed(args.run_id)
    traj, test_traj, runtime = run_random_search(iterations=args.iter, verbose=args.verbose)
    valid_scores, test_scores = calc_regrets(traj, test_traj)
    save_json(valid_scores, test_scores, runtime, output_path, args.run_id)

else:  # for multiple runs
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if not args.fix_seed:
            np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        traj, test_traj, runtime = run_random_search(iterations=args.iter, verbose=args.verbose)
        valid_scores, test_scores = calc_regrets(traj, test_traj)
        save_json(valid_scores, test_scores, runtime, output_path, run_id)

save_configspace(cs, output_path)
