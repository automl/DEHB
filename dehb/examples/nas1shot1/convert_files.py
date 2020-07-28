'''Script to convert .obj files dumped by NAS-1shot1 scripts to .json for plotting

Expected directory structure:

TODO: tree
'''

import os
import json
import pickle
import numpy as np

import argparse

import sys
sys.path.append(os.path.join(os.getcwd(), '../nasbench-1shot1/'))


from optimizers.utils import Model, Architecture

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3


class DotAccess():
    def __init__(self, valid, info, test):
        self.valid = valid
        self.info = info
        self.test = test


def process_and_save(all_runs):
    global y_star_valid, y_star_test
    valid_incumbents = []
    runtimes = []
    test_incumbents = []
    inc = np.inf
    test_regret = 1

    for k in range(len(all_runs)):
        print('Iteration {:<3}/{:<3}'.format(k+1, len(all_runs)), end="\r", flush=True)
        regret = all_runs[k].valid - y_star_valid
        # Update test regret only when incumbent changed by validation regret
        if regret <= inc:
            inc = regret
            test_regret = all_runs[k].test - y_star_test
        valid_incumbents.append(inc)
        test_incumbents.append(test_regret)
        runtimes.append(all_runs[k].info)
    runtimes = np.cumsum(runtimes).tolist()
    return valid_incumbents, runtimes, test_incumbents


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='experiments/discrete_optimizers',
		    type=str, help='path to files')
parser.add_argument('--algo', default='BOHB', type=str, nargs='?')
parser.add_argument('--ssp', default=1, type=int, nargs='?')
parser.add_argument('--n_runs', default=500, type=int, nargs='?',
                    help='number of runs to plot data for')

args = parser.parse_args()
n_runs = args.n_runs
algo = args.algo
ssp = args.ssp
path = args.path

search_space = eval('SearchSpace{}()'.format(ssp))
y_star_valid, y_star_test, inc_config = (search_space.valid_min_error,
                                         search_space.test_min_error, None)

for i in range(n_runs):
    print("\nRun {}".format(i))
    if algo in ['HB', 'BOHB', 'TPE', 'SMAC']:
        with open(os.path.join(path, '{}/{}/'
                  'algo_{}_{}_ssp_{}_seed_0.obj'.format(algo, ssp, algo, i, ssp)), 'rb') as f:
            res = pickle.load(f)
    elif algo in ['RE', 'RS']:
        with open(os.path.join(path, '{}/{}/'
                  'algo_{}_None_ssp_{}_seed_{}.obj'.format(algo, ssp, algo, ssp, i)), 'rb') as f:
            res = pickle.load(f)
    elif 'dehb' in algo:
        with open(os.path.join(path, '{}/{}/'
                  'DEHB_{}_ssp_{}_seed_{}.obj'.format(algo, ssp, i, ssp, i)), 'rb') as f:
            res = pickle.load(f)
    elif 'de' in algo:
        with open(os.path.join(path, '{}/{}/'
                  'DE_{}_ssp_{}_seed_{}.obj'.format(algo, ssp, i, ssp, i)), 'rb') as f:
            res = pickle.load(f)
    elif 'scipy' in algo:
        with open(os.path.join(path, '{}/{}/'
                  'DE_{}_ssp_{}_seed_{}.obj'.format(algo, ssp, i, ssp, i)), 'rb') as f:
            res = pickle.load(f)

    all_runs = []
    for j in range(len(res)):
        all_runs.append(DotAccess(valid = 1 - res[j].validation_accuracy,
                                  info  = res[j].training_time,
                                  test  = 1 - res[j].test_accuracy))
    valid_incumbents, runtimes, test_incumbents = process_and_save(all_runs)
    with open(os.path.join(path, '{}/{}/'
              'run_{}.json'.format(algo, ssp, i)), 'w') as f:
        json.dump({'runtime': runtimes, 'regret_validation': valid_incumbents,
                   'regret_test': test_incumbents}, f)
