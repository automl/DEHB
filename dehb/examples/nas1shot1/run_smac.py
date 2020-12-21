# Slightly modified version of the script from:
# https://github.com/automl/nasbench-1shot1/blob/master/optimizers/smac/run_smac.py

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nasbench/'))
sys.path.append(os.path.join(os.getcwd(), '../nasbench-1shot1/'))

import json
import pickle
import argparse
import logging
import numpy as np
import ConfigSpace
logging.basicConfig(level=logging.INFO)

from smac.scenario.scenario import Scenario

from nasbench import api

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from nasbench_analysis.utils import INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3

sys.path.append('dehb/examples/')
from utils import util

sys.path.append(os.path.join(os.getcwd(), '../HpBandSter/icml_2018_experiments/experiments/workers'))
from base_worker import BaseWorker


def convert_to_json(results, y_star_valid, y_star_test):
    res = {}
    res['regret_validation'] = np.array(results['losses'] - y_star_valid).tolist()
    res['regret_test'] = np.array(results['test_losses'] - y_star_test).tolist()
    res['runtime'] = np.array(results['cummulative_cost']).tolist()
    return res


class NAS1shot1Worker(BaseWorker):
    def __init__(self, benchmark, **kwargs):
        super().__init__(max_budget=108, **kwargs)
        self.b = benchmark
        self.cs = self.b.get_configuration_space()

    def compute(self, config, **kwargs):
        c = ConfigSpace.Configuration(self.cs, values=config)
        y, cost = self.b.objective_function(nasbench, c, budget=108)
        y_test, _ = self.b.objective_function_test(nasbench, c, budget=108)

        return ({
            'loss': 1 - float(y),
            'info': {'cost': float(cost), 'test_loss': 1 - float(y_test)}
        })


parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--search_space', default=None, type=str, nargs='?',
                    help='specifies the benchmark')
parser.add_argument('--n_iters', default=280, type=int, nargs='?',
                    help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./experiments", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir',
                    default="../nasbench-1shot1/nasbench_analysis/nasbench_data/108_e/nasbench_only108.tfrecord",
                    type=str, nargs='?', help='specifies the path to the nasbench data')
parser.add_argument('--seed', default=0, type=int,
                    help='random seed')
parser.add_argument('--n_repetitions', default=500, type=int,
                    help='number of repetitions')
args = parser.parse_args()

nasbench = api.NASBench(args.data_dir)

output_path = os.path.join(args.output_path, "SMAC")
os.makedirs(os.path.join(output_path), exist_ok=True)
args.working_directory = output_path
args.method = "smac"
args.num_iterations = args.n_iters


if args.search_space is None:
    spaces = [1, 2, 3]
else:
    spaces = [int(args.search_space)]

for space in spaces:
    print('##### Search Space {} #####'.format(space))
    print('##### Seed {} #####'.format(args.seed))
    np.random.seed(args.seed)
    run_id = args.run_id
    y_star_valid, y_star_test, inc_config = (search_space.valid_min_error,
                                             search_space.test_min_error, None)

    search_space = eval('SearchSpace{}()'.format(space))
    cs = search_space.get_configuration_space()

    worker = NAS1shot1Worker(
        benchmark=search_space, measure_test_loss=False, run_id=run_id, max_budget=108
    )
    result = util.run_experiment(args, worker, output_path, smac_deterministic=False)
    with open(os.path.join(output_path, 'smac_run_{}.pkl'.format(run_id)), "rb") as f:
        result = pickle.load(f)
    fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
    json.dump(convert_to_json(result, y_star_valid, y_star_test), fh)
    fh.close()
    os.remove(os.path.join(output_path, 'smac_run_{}.pkl'.format(run_id)))
