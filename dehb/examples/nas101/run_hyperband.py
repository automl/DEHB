# Slightly modified version of:
# https://github.com/automl/nas_benchmarks/blob/development/experiment_scripts/run_hyperband.py

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks/'))
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks-development/'))

import ConfigSpace
import json
import argparse
import logging
logging.basicConfig(level=logging.ERROR)

from hpbandster.optimizers.hyperband import HyperBand
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C

parser = argparse.ArgumentParser()
parser.add_argument('--runs', default=1, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--benchmark', default="nas_cifar10", type=str, nargs='?',
                    help='specifies the benchmark')
parser.add_argument('--n_iters', default=100, type=int, nargs='?',
                    help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../tabular_benchmarks/fcnet_tabular_benchmarks/",
                    type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--eta', default=3, type=int, nargs='?',
                    help='eta parameter of successive halving')

args = parser.parse_args()

if args.benchmark == "nas_cifar10a":
    min_budget = 4
    max_budget = 108
    b = NASCifar10A(data_dir=args.data_dir)

elif args.benchmark == "nas_cifar10b":
    b = NASCifar10B(data_dir=args.data_dir)
    min_budget = 4
    max_budget = 108

elif args.benchmark == "nas_cifar10c":
    b = NASCifar10C(data_dir=args.data_dir)
    min_budget = 4
    max_budget = 108

elif args.benchmark == "protein_structure":
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)
    min_budget = 3
    max_budget = 100

elif args.benchmark == "slice_localization":
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)
    min_budget = 3
    max_budget = 100

elif args.benchmark == "naval_propulsion":
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)
    min_budget = 3
    max_budget = 100

elif args.benchmark == "parkinsons_telemonitoring":
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)
    min_budget = 3
    max_budget = 100

output_path = os.path.join(args.output_path, "hyperband")
os.makedirs(os.path.join(output_path), exist_ok=True)

cs = b.get_configuration_space()


class MyWorker(Worker):
    def compute(self, config, budget, *args, **kwargs):
        c = ConfigSpace.Configuration(cs, values=config)
        y, cost = b.objective_function(c, budget=int(budget))
        return ({
            'loss': float(y),
            'info': float(cost)})

runs = args.runs
for run_id in range(runs):
    print("Run {:>3}/{:>3}".format(run_id+1, runs))

    # hb_run_id = '0'
    hb_run_id = run_id

    NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
    ns_host, ns_port = NS.start()

    num_workers = 1

    workers = []
    for i in range(num_workers):
        w = MyWorker(nameserver=ns_host, nameserver_port=ns_port,
                     run_id=hb_run_id,
                     id=i)
        w.run(background=True)
        workers.append(w)

    HB = HyperBand(configspace=cs,
                   run_id=hb_run_id,
                   eta=args.eta, min_budget=min_budget, max_budget=max_budget,
                   nameserver=ns_host,
                   nameserver_port=ns_port,
                   ping_interval=10)

    results = HB.run(args.n_iters, min_n_workers=num_workers)

    HB.shutdown(shutdown_workers=True)
    NS.shutdown()

    if 'cifar' in args.benchmark:
        res = b.get_results(ignore_invalid_configs=True)
    else:
        res = b.get_results()

    fh = open(os.path.join(output_path, 'run_%d.json' % run_id), 'w')
    json.dump(res, fh)
    fh.close()
    print("Run saved. Resetting...")
    b.reset_tracker()
