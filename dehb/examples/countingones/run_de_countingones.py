import os
import sys
import json
import pickle
import argparse
import numpy as np

import ConfigSpace

from hpolib.benchmarks.synthetic_functions.counting_ones import CountingOnes

from dehb import DE, AsyncDE


# Common objective function for DE & DEHB representing SVM Surrogates benchmark
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


def calc_regrets(history):
    global de, b, cs
    d = len(cs.get_hyperparameters())
    valid_scores = []
    test_scores = []
    test_regret = 1
    valid_regret = 1
    inc = np.inf
    for i in range(len(history)):
        valid_regret = (history[i][1] + d) / d
        if valid_regret < inc:
            inc = valid_regret
            config = de.vector_to_configspace(history[i][0])
            res = b.objective_function_test(config)
            test_regret = (res['function_value'] + d) / d
        test_scores.append(test_regret)
        valid_scores.append(inc)
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
parser.add_argument('--gens', default=100, type=int, nargs='?',
                    help='(iterations) number of generations for DE to evolve')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--pop_size', default=20, type=int, nargs='?', help='population size')
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
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default=None, type=str, nargs='?',
                    help='name of folder where files will be dumped')
parser.add_argument('--async', default=None, type=str, nargs='?',
                    choices=['deferred', 'immediate', 'random', 'worst'],
                    help='type of Asynchronous DE')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False

dim_folder = "{}+{}".format(args.n_cont, args.n_cat)
if args.async is None:
    folder = "{}/de_pop{}".format(dim_folder, args.pop_size)
else:
    folder = "{}/ade_{}_pop{}".format(dim_folder, args.async, args.pop_size)

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


# Initializing DE object
if args.async is None:
    de = DE(cs=cs, dimensions=dimensions, f=f, pop_size=args.pop_size,
            mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
            strategy=args.strategy, budget=max_budget)
else:
    de = AsyncDE(cs=cs, dimensions=dimensions, f=f, pop_size=args.pop_size,
                 mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
                 strategy=args.strategy, budget=max_budget, async_strategy=args.async)


if args.runs is None:  # for a single run
    if not args.fix_seed:
        np.random.seed(0)
    # Running DE iterations
    traj, runtime, history = de.run(generations=args.gens, verbose=args.verbose)
    valid_scores, test_scores = calc_regrets(history)

    save_json(valid_scores, test_scores, runtime, output_path, args.run_id)

else:  # for multiple runs
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if not args.fix_seed:
            np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        # Running DE iterations
        traj, runtime, history = de.run(generations=args.gens, verbose=args.verbose)
        valid_scores, test_scores = calc_regrets(history)

        save_json(valid_scores, test_scores, runtime, output_path, run_id)

        if args.verbose:
            print("Run saved. Resetting...")
        # essential step to not accumulate consecutive runs

        de.reset()

save_configspace(cs, output_path)
