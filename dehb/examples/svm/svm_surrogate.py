import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../HpBandSter/icml_2018_experiments/experiments'))
sys.path.append(os.path.join(os.getcwd(), 'dehb/examples/'))

import argparse

from workers.svm_surrogate import SVMSurrogateWorker as Worker
from utils import util

# deactivate debug output for faster experiments
import logging
logging.basicConfig(level=logging.ERROR)


################################################################################
#                    Benchmark specific stuff
################################################################################


parser = argparse.ArgumentParser(description='Run different optimizers on the SVM Surrogates '
                                             'problem.', conflict_handler='resolve')
parser = util.standard_parser_args(parser)

# add benchmark specific arguments
parser.add_argument('--dest_dir', type=str, help='the destination directory', default='./results/')
parser.add_argument('--num_iterations', type=int, default=4,
                    help='number of Hyperband iterations performed.')
parser.add_argument('--run_id', type=str, default=0)
parser.add_argument('--runs', type=int, default=None)
parser.add_argument('--method', type=str, default='randomsearch',
                    help='Possible choices: randomsearch, bohb, hyperband, tpe, smac, h2bo')
parser.add_argument('--surrogate_path', type=str, default=None,
                    help='Path to the pickled surrogate models. If None, HPOlib2 will '
                         'automatically download the surrogates to the .hpolib directory '
                         'in your home directory.')
parser.add_argument('--min_budget', type=float, default=1/512,
                    help='Smallest fraction of the full dataset that is used.')
parser.add_argument('--max_budget', type=float, default=1,
                    help='Largest fraction of the full dataset that is used.')
parser.add_argument('--folder', type=float, default=None,
                    help='folder to dump output files')

args = parser.parse_args()

if args.folder is None:
    folder = args.method

# if args.runs is None:

# this is a synthetic benchmark, so we will use the run_id to separate the independent runs
worker = Worker(surrogate_path=args.surrogate_path, measure_test_loss=True, run_id=args.run_id)

# directory where the results are stored
dest_dir = os.path.join(args.dest_dir, folder)
args.working_directory = dest_dir

# SMAC can be informed whether the objective is deterministic or not
smac_deterministic = True

# run experiment
result = util.run_experiment(args, worker, dest_dir, smac_deterministic)
