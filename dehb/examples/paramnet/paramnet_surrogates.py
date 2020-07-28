import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../HpBandSter/icml_2018_experiments/experiments'))
sys.path.append(os.path.join(os.getcwd(), 'dehb/examples/paramnet'))
sys.path.append(os.path.join(os.getcwd(), 'dehb/examples/'))

import argparse
import numpy as np

from worker_paramnet_surrogates import ParamNetSurrogateWorker as Worker
from utils import util


# deactivate debug output for faster experiments
import logging
logging.basicConfig(level=logging.ERROR)


################################################################################
#                    Benchmark specific stuff
################################################################################
parser = argparse.ArgumentParser(description='Run different optimizers on the '
                                             'CountingOnes problem.', conflict_handler='resolve')
parser = util.standard_parser_args(parser)


# add benchmark specific arguments
parser.add_argument('--dest_dir', type=str, help='the destination directory', default='./results/')
parser.add_argument('--num_iterations', type=int, default=4,
                    help='number of Hyperband iterations performed.')
parser.add_argument('--run_id', type=str, default=0)
parser.add_argument('--runs', type=int, default=None)
parser.add_argument('--method', type=str, default='randomsearch',
                    help='Possible choices: randomsearch, bohb, hyperband, tpe, smac')
parser.add_argument('--dataset', default='mnist', help="name of the dataset used",
                    choices=['adult', 'higgs', 'letter', 'mnist', 'optdigits', 'poker'])
parser.add_argument('--surrogate_path', type=str, default=None,
                    help='path to the pickled surrogate models')
parser.add_argument('--folder', type=float, default=None,
                    help='folder to dump output files')

args = parser.parse_args()

if args.folder is None:
    folder = args.method

# this is a synthetic benchmark, so we will use the run_id to separate the independent runs
worker = Worker(dataset=args.dataset, surrogate_path=args.surrogate_path,
                measure_test_loss=False, run_id=args.run_id)

args.min_budget, args.max_budget = worker.budgets[args.dataset]

# directory where the results are stored
dest_dir = os.path.join(args.dest_dir, args.dataset, folder)
args.working_directory = dest_dir

# SMAC can be informed whether the objective is deterministic or not
smac_deterministic = True

# run experiment
result = util.run_experiment(args, worker, dest_dir, smac_deterministic)
