# Slightly modified version of the script from:
# https://github.com/automl/nasbench-1shot1/blob/master/optimizers/hyperband/run_hyperband.py


import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nasbench/'))
sys.path.append(os.path.join(os.getcwd(), '../nasbench-1shot1/'))


import pickle
import argparse
import logging
import numpy as np
import ConfigSpace
logging.basicConfig(level=logging.INFO)

import hpbandster.core.nameserver as hpns
from hpbandster.optimizers.hyperband import HyperBand
from hpbandster.core.worker import Worker
from nasbench import api

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from nasbench_analysis.utils import INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3

from IPython import embed

class MyWorker(Worker):
    def compute(self, config, budget, *args, **kwargs):
        c = ConfigSpace.Configuration(cs, values=config)
        y, cost = search_space.objective_function(nasbench, c, budget=int(budget))
        return ({
            'loss': 1 - float(y),
            'info': float(cost)})


parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--search_space', default=None, type=str, nargs='?',
                    help='specifies the benchmark')
parser.add_argument('--n_iters', default=280, type=int, nargs='?',
                    help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./experiments", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', type=str, nargs='?',
                    default="../nasbench-1shot1/nasbench_analysis/nasbench_data/"
                            "108_e/nasbench_full.tfrecord",
                    help='specifies the path to the nasbench data')
parser.add_argument('--seed', default=0, type=int,
                    help='random seed')
parser.add_argument('--n_repetitions', default=500, type=int,
                    help='number of repetitions')
parser.add_argument('--eta', default=3, type=int, nargs='?',
                    help='eta parameter of successive halving')
parser.add_argument('--min_budget', default=4, type=int, nargs='?',
                    help='minimum budget')
parser.add_argument('--max_budget', default=108, type=int, nargs='?',
                    help='maximum budget')

args = parser.parse_args()

min_budget = args.min_budget
max_budget = args.max_budget
nasbench = api.NASBench(args.data_dir)

output_path = os.path.join(args.output_path, "discrete_optimizers", 'HB')
os.makedirs(os.path.join(output_path), exist_ok=True)

if args.search_space is None:
    spaces = [1, 2, 3]
else:
    spaces = [int(args.search_space)]

#embed()

for space in spaces:
    print('##### Search Space {} #####'.format(space))
    search_space = eval('SearchSpace{}()'.format(space))
    cs = search_space.get_configuration_space()

    #for seed in range(args.n_repetitions):
    print('##### Seed {} #####'.format(args.seed))
    # Set random_seed
    np.random.seed(args.seed)

    hb_run_id = '0'

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
                   eta=args.eta,
                   min_budget=min_budget,
                   max_budget=max_budget,
                   nameserver=ns_host,
                   nameserver_port=ns_port,
                   ping_interval=10)

    history = HB.run(args.n_iters, min_n_workers=num_workers)

    HB.shutdown(shutdown_workers=True)
    NS.shutdown()

    fh = open(os.path.join(output_path,
                           'algo_{}_{}_ssp_{}_seed_{}.obj'.format('HB',
                                                                  args.run_id,
                                                                  space,
                                                                  args.seed)), 'wb')
    pickle.dump(search_space.run_history, fh)
    fh.close()

    print(min([1 - arch.test_accuracy - search_space.test_min_error for
               arch in search_space.run_history]))
