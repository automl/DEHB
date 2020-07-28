'''Runs DE on Cartpole surrogate
'''

import os
import sys
import json
import pickle
import argparse
import numpy as np
import ConfigSpace

from hpolib.benchmarks.rl.cartpole import CartpoleReduced as surrogate

from dehb import DE, AsyncDE


# Common objective function for DE & DEHB representing Cartpole RL surrogates
def f(config, budget=None):
    global max_budget, b
    if budget is not None:
        res = b.objective_function(config, budget=int(budget))
    else:
        res = b.objective_function(config)
    fitness = res['function_value']
    cost = res['cost']
    return fitness, cost


def calc_regrets(history):
    global de, b, cs
    valid_scores = []
    test_scores = []
    test_regret = 1
    valid_regret = 1
    inc = np.inf
    for i in range(len(history)):
        valid_regret = history[i][1]
        if valid_regret < inc:
            inc = valid_regret
            config = de.vector_to_configspace(history[i][0])
            res = b.objective_function_test(config)
            test_regret = res['function_value']
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
    fh = open(os.path.join(output_path, '{}.pkl'.format(filename)), 'wb')
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
parser.add_argument('--gens', default=1, type=int, nargs='?',
                    help='number of generations for DE to evolve')
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
parser.add_argument('--pop_size', default=20, type=int, nargs='?',
                    help='population size')
parser.add_argument('--max_budget', default=9, type=int, nargs='?',
                    help='maximum budget')
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
max_budget = args.max_budget

# Directory where files will be written
if args.async is None:
    folder = "de_pop{}".format(args.pop_size)
else:
    folder = "ade_{}_pop{}".format(args.async, args.pop_size)
output_path = os.path.join(args.output_path, folder)
os.makedirs(output_path, exist_ok=True)

# Parameter space to be used by DE
b = surrogate()
cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())


# Initializing DE object
if args.async is None:
    de = DE(cs=cs, dimensions=dimensions, f=f, pop_size=args.pop_size,
            mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
            strategy=args.strategy, budget=args.max_budget)
else:
    de = AsyncDE(cs=cs, dimensions=dimensions, f=f, pop_size=args.pop_size,
                 mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
                 strategy=args.strategy, budget=args.max_budget, async_strategy=args.async)


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
        print("Run saved. Resetting...")
        # essential step to not accumulate consecutive runs
        de.reset()

save_configspace(cs, output_path)
