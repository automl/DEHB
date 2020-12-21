# Slightly modified version of:
# https://github.com/automl/nas_benchmarks/blob/development/experiment_scripts/run_smac.py

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks/'))
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks-development/'))

import json
import argparse

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO as SMAC

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C

parser = argparse.ArgumentParser()

parser.add_argument('--start', default=0, type=int, nargs='?',
                    help='unique number to start the run_id')
parser.add_argument('--runs', default=1, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--benchmark', default="nas_cifar10a", type=str, nargs='?',
                    help='specifies the benchmark')
parser.add_argument('--n_iters', default=5, type=int, nargs='?',
                    help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../nas_benchmarks-development/"
                                          "tabular_benchmarks/fcnet_tabular_benchmarks/",
                    type=str, nargs='?', help='specifies the path to the tabular data')
args = parser.parse_args()

if args.benchmark == "nas_cifar10a":
    b = NASCifar10A(data_dir=args.data_dir, multi_fidelity=False)

elif args.benchmark == "nas_cifar10b":
    b = NASCifar10B(data_dir=args.data_dir, multi_fidelity=False)

elif args.benchmark == "nas_cifar10c":
    b = NASCifar10C(data_dir=args.data_dir, multi_fidelity=False)

elif args.benchmark == "protein_structure":
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)

elif args.benchmark == "slice_localization":
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)

elif args.benchmark == "naval_propulsion":
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)

elif args.benchmark == "parkinsons_telemonitoring":
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)

output_path = os.path.join(args.output_path, "smac")
os.makedirs(os.path.join(output_path), exist_ok=True)

cs = b.get_configuration_space()

scenario = Scenario({"run_obj": "quality",
                     "runcount-limit": args.n_iters,
                     "cs": cs,
                     "deterministic": "false",
                     "initial_incumbent": "RANDOM",
                     "output_dir": ""})


def objective_function(config, **kwargs):
    y, c = b.objective_function(config)
    return float(y)


start = args.start
runs = args.runs
assert runs > start
for run_id in range(start, runs):
    print("Run {:>3}/{:>3}".format(run_id+1, runs))

    smac = SMAC(scenario=scenario, tae_runner=objective_function)
    smac.optimize()

    if 'cifar' in args.benchmark:
        res = b.get_results(ignore_invalid_configs=True)
    else:
        res = b.get_results()

    fh = open(os.path.join(output_path, 'run_%d.json' % run_id), 'w')
    json.dump(res, fh)
    fh.close()
    print("Run saved. Resetting...")
    b.reset_tracker()
