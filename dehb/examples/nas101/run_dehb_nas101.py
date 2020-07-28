'''Runs DEHB on NAS-Bench-101 and NAS-HPO-Bench
'''

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks/'))
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks-development/'))

import json
import pickle
import argparse
import numpy as np

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C

from dehb import DE
from dehb import DEHB, DEHB_0, DEHB_1, DEHB_2, DEHB_3


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


# Common objective function for DE representing NAS-Bench-101 & NAS-HPO-Bench
def f(config, budget=None):
    if budget is not None:
        fitness, cost = b.objective_function(config, budget=int(budget))
    else:
        fitness, cost = b.objective_function(config)
    return fitness, cost


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
choices = ["protein_structure", "slice_localization", "naval_propulsion",
           "parkinsons_telemonitoring", "nas_cifar10a", "nas_cifar10b", "nas_cifar10c"]
parser.add_argument('--benchmark', default="protein_structure", type=str,
                    help="specify the benchmark to run on from among {}".format(choices))
parser.add_argument('--gens', default=1, type=int, nargs='?',
                    help='(iterations) number of generations for DE to evolve')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../nas_benchmarks-development/"
                                          "tabular_benchmarks/fcnet_tabular_benchmarks/",
                    type=str, nargs='?', help='specifies the path to the tabular data')
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
parser.add_argument('--eta', default=3, type=int, nargs='?',
                    help='eta for Successive Halving')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default=None, type=str, nargs='?',
                    help='name of folder where files will be dumped')
parser.add_argument('--version', default=None, type=str, nargs='?',
                    help='version of DEHB to run')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False

if args.benchmark == "nas_cifar10a": # NAS-Bench-101
    min_budget = 4
    max_budget = 108
    b = NASCifar10A(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "nas_cifar10b": # NAS-Bench-101
    min_budget = 4
    max_budget = 108
    b = NASCifar10B(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "nas_cifar10c": # NAS-Bench-101
    min_budget = 4
    max_budget = 108
    b = NASCifar10C(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "protein_structure": # NAS-HPO-Bench
    min_budget = 3
    max_budget = 100
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "slice_localization": # NAS-HPO-Bench
    min_budget = 3
    max_budget = 100
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "naval_propulsion": # NAS-HPO-Bench
    min_budget = 3
    max_budget = 100
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "parkinsons_telemonitoring": # NAS-HPO-Bench
    min_budget = 3
    max_budget = 100
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

# Parameter space to be used by DE
cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

if args.folder is None:
    folder = "dehb"
    if args.version is not None:
        folder = "dehb_v{}".format(args.version)
else:
    folder = args.folder

output_path = os.path.join(args.output_path, folder)
os.makedirs(output_path, exist_ok=True)

dehbs = {None: DEHB, "0": DEHB_0, "1": DEHB_1, "2": DEHB_2, "3": DEHB_3}
DEHB = dehbs[args.version]

# Initializing DEHB object
dehb = DEHB(cs=cs, dimensions=dimensions, f=f, strategy=args.strategy,
            mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
            eta=args.eta, min_budget=min_budget, max_budget=max_budget,
            generations=args.gens)

if args.runs is None:  # for a single run
    if not args.fix_seed:
        np.random.seed(0)
    # Running DEHB iterations
    traj, runtime, history = dehb.run(iterations=args.iter, verbose=args.verbose)
    if 'cifar' in args.benchmark:
        res = b.get_results(ignore_invalid_configs=True)
    else:
        res = b.get_results()
    fh = open(os.path.join(output_path, 'run_{}.json'.format(args.run_id)), 'w')
    json.dump(res, fh)
    fh.close()
else:  # for multiple runs
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if not args.fix_seed:
            np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        # Running DEHB iterations
        traj, runtime, history = dehb.run(iterations=args.iter, verbose=args.verbose)
        if 'cifar' in args.benchmark:
            res = b.get_results(ignore_invalid_configs=True)
        else:
            res = b.get_results()
        fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
        json.dump(res, fh)
        fh.close()
        if args.verbose:
            print("Run saved. Resetting...")
        # essential step to not accumulate consecutive runs
        dehb.reset()
        b.reset_tracker()

save_configspace(cs, output_path)
