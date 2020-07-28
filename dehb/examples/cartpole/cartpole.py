import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../HpBandSter/icml_2018_experiments/experiments'))
sys.path.append(os.path.join(os.getcwd(), 'dehb/examples/'))

import argparse
import numpy as np

# catch missing dependencies here
try:
	import tensorforce
	import gym
except ImportError:
	raise ImportError("You need to install 'tensorforce' and the OpenAI 'gym' package"
                      " for this benchmark!")
except:
	raise

from workers.cartpole import CartpoleReducedWorker as Worker
import util

# deactivate debug output for faster experiments
import logging
logging.basicConfig(level=logging.DEBUG)


################################################################################
#                    Benchmark specific stuff
################################################################################


parser = argparse.ArgumentParser(conflict_handler='resolve',
                                 description='Run different optimizers on the '
                                             'CountingOnes problem.')
parser = util.standard_parser_args(parser)


# add benchmark specific arguments
parser.add_argument('--min_budget', type=int, default=1,
                    help='Minimum number of independent runs to estimate mean loss.')
parser.add_argument('--max_budget', type=int, default=9,
                    help='Maximum number of independent runs to estimate mean loss.')
parser.add_argument('--num_iterations', type=int, default=16,
                    help='number of Hyperband iterations performed.')

args = parser.parse_args()

# this is a synthetic benchmark, so we will use the run_id to separate the independent runs
worker = Worker(measure_test_loss=False, run_id=args.run_id)

# directory where the results are stored
dest_dir = os.path.join(args.dest_dir, args.method)
args.working_directory = dest_dir

# SMAC can be informed whether the objective is deterministic or not
smac_deterministic = True

# run experiment
# this is a rather expensive benchmark, so we store all the results of each run
result = util.run_experiment(args, worker, dest_dir, smac_deterministic, store_all_runs=True)
