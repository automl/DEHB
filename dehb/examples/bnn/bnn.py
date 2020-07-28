import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../HpBandSter/icml_2018_experiments/experiments'))
sys.path.append(os.path.join(os.getcwd(), 'dehb/examples/'))

import argparse
import numpy as np

from workers.bnn import BNNWorker as Worker
import util

# deactivate debug output for faster experiments
import logging
logging.basicConfig(level=logging.DEBUG)


################################################################################
#                    Benchmark specific stuff
################################################################################
parser = argparse.ArgumentParser(conflict_handler='resolve',
                                 description='Run different optimizers to optimize BNNs on '
                                             'different datasets.')
parser = util.standard_parser_args(parser)


# add benchmark specific arguments
parser.add_argument('--dataset', help="name of the dataset used", default='bostonhousing',
                    choices=['toyfunction', 'bostonhousing', 'proteinstructure', 'yearprediction'])
parser.add_argument('--min_budget', type=int, default=500,
                    help='Minimum number of MCMC steps used to draw samples for the BNN.')
parser.add_argument('--max_budget', type=int, default=10000,
                    help='Maximum number of MCMC steps used to draw samples for the BNN.')

args = parser.parse_args()


# this is a synthetic benchmark, so we will use the run_id to separate the independent runs
worker = Worker(dataset=args.dataset, measure_test_loss=False, run_id=args.run_id,
                max_budget=args.max_budget)

# directory where the results are stored
dest_dir = os.path.join(args.dest_dir, args.dataset, args.method)
args.working_directory = dest_dir

# SMAC can be informed whether the objective is deterministic or not
smac_deterministic = True

# run experiment
result = util.run_experiment(args, worker, dest_dir, smac_deterministic, store_all_runs=True)
print(result.get_all_runs())
