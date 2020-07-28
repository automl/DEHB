# Slightly modified version of the script from:
# https://github.com/automl/nasbench-1shot1/blob/master/optimizers/regularized_evolution/run_regularized_evolution.py

"""
Regularized evolution as described in:
Real, E., Aggarwal, A., Huang, Y., and Le, Q. V.
Regularized Evolution for Image Classifier Architecture Search.
In Proceedings of the Conference on Artificial Intelligence (AAAI’19)

The code is based one the original regularized evolution open-source implementation:
https://colab.research.google.com/github/google-research/google-research/blob/master/evolution/regularized_evolution_algorithm/regularized_evolution.ipynb

NOTE: This script has certain deviations from the original code owing to the search space of the benchmarks used:
1) The fitness function is not accuracy but error and hence the negative error is being maximized.
2) The architecture is a ConfigSpace object that defines the model architecture parameters.

Adaptions were made to make it compatible with the search spaces.
"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nasbench/'))
sys.path.append(os.path.join(os.getcwd(), '../nasbench-1shot1/'))

import copy
import pickle
import random
import argparse
import collections

import numpy as np
from nasbench import api

from optimizers.utils import Model, Architecture
from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from nasbench_analysis.utils import upscale_to_nasbench_format, INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3


def train_and_eval(config):
    adjacency_matrix, node_list = config.adjacency_matrix, config.node_list
    if type(search_space) == SearchSpace1 or type(search_space) == SearchSpace2:
        # Fill up adjacency matrix and node list with entries for unused nodes
        adjacency_matrix = upscale_to_nasbench_format(adjacency_matrix)
        node_list = [INPUT, *node_list, CONV1X1, OUTPUT]
    else:
        node_list = [INPUT, *node_list, OUTPUT]
    adjacency_list = adjacency_matrix.astype(np.int).tolist()
    model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
    nasbench_data = nasbench.query(model_spec)
    return nasbench_data['validation_accuracy'], nasbench_data['test_accuracy'], nasbench_data['training_time']


def random_architecture():
    adjacency_matrix, node_list = search_space.sample(with_loose_ends=True, upscale=False)
    architecture = Architecture(adjacency_matrix=adjacency_matrix, node_list=node_list)
    return architecture


def mutate_arch(parent_arch):
    # Choose one of the three mutations
    mutation = np.random.choice(['identity', 'hidden_state_mutation', 'op_mutation'])

    adjacency_matrix, node_list = copy.deepcopy(parent_arch.adjacency_matrix), copy.deepcopy(parent_arch.node_list)
    if mutation == 'identity':
        return Architecture(adjacency_matrix=adjacency_matrix, node_list=node_list)
    elif mutation == 'hidden_state_mutation':
        # Pick one of the intermediate nodes in the graph (neither input nor output)
        if type(search_space) == SearchSpace1:
            # Node 1 has now choice for node 2 as it always has 2 parents
            low = 3
        else:
            low = 2
        random_node = np.random.randint(low=low, high=adjacency_matrix.shape[-1])
        # Pick one of the parents of the node
        parent_of_node_to_modify = np.random.choice(adjacency_matrix[:random_node, random_node].nonzero()[0])
        # Select a new parent for this node, (needs to be different from previous parent)
        new_parent_of_node = np.random.choice(np.argwhere(adjacency_matrix[:random_node, random_node] == 0).flatten())
        # Remove old parent from child
        adjacency_matrix[parent_of_node_to_modify, random_node] = 0
        # Add new parent to child
        adjacency_matrix[new_parent_of_node, random_node] = 1
        # Create new child config
        return Architecture(adjacency_matrix=adjacency_matrix, node_list=node_list)
    else:  # op_mutation
        OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
        op_idx_to_change = np.random.randint(len(node_list))
        # Remove current op on selected idx (because we want a new op)
        OPS.remove(node_list[op_idx_to_change])
        # Select one of the remaining ops
        new_op = np.random.choice(OPS)
        node_list[op_idx_to_change] = new_op
        return Architecture(adjacency_matrix=adjacency_matrix, node_list=node_list)


def regularized_evolution(cycles, population_size, sample_size):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    Args:
      cycles: the number of cycles the algorithm should run for.
      population_size: the number of individuals to keep in the population.
      sample_size: the number of individuals that should participate in each
          tournament.

    Returns:
      history: a list of `Model` instances, representing all the models computed
          during the evolution experiment.
    """
    population = collections.deque()
    history = []  # Not used by the algorithm, only used to report results.

    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture()
        model.validation_accuracy, model.test_accuracy, model.training_time = train_and_eval(model.arch)
        population.append(model)
        history.append(model)

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = []
        while len(sample) < sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.validation_accuracy)

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch)
        child.validation_accuracy, child.test_accuracy, child.training_time = train_and_eval(child.arch)
        population.append(child)
        history.append(child)

        # Remove the oldest model.
        population.popleft()

    return history


def random_search(cycles):
    history = []
    for i in range(cycles):
        model = Model()
        model.arch = random_architecture()
        model.validation_accuracy, model.test_accuracy, model.training_time = train_and_eval(model.arch)
        history.append(model)
    return history


parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--algorithm', default=None, type=str)
parser.add_argument('--search_space', default=None, type=str, nargs='?', help='specifies the benchmark')
parser.add_argument('--n_iters', default=280, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./experiments", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', type=str, nargs='?',
                    default="../nasbench-1shot1/nasbench_analysis/nasbench_data/"
                            "108_e/nasbench_full.tfrecord",
                    help='specifies the path to the nasbench data')
parser.add_argument('--pop_size', default=100, type=int, nargs='?', help='population size')
parser.add_argument('--sample_size', default=10, type=int, nargs='?', help='sample_size')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--n_repetitions', default=500, type=int, help='number of repetitions')

args = parser.parse_args()
nasbench = api.NASBench(args.data_dir)

if args.search_space is None:
    spaces = [1, 2, 3]
else:
    spaces = [int(args.search_space)]

if args.algorithm is None:
    algos = ['RE', 'RS']
else:
    algos = [args.algorithm]


for space in spaces:
    search_space = eval('SearchSpace{}()'.format(space))
    for alg in algos:
        print("##### Algorithm {} #####".format(alg))
        for seed in range(args.n_repetitions):
            print("##### Seed {} #####".format(seed))
            np.random.seed(seed)
            output_path = os.path.join(args.output_path, "discrete_optimizers", alg)
            os.makedirs(os.path.join(output_path), exist_ok=True)

            # Set random_seed
            if alg == 'RE':
                history = regularized_evolution(
                    cycles=args.n_iters, population_size=args.pop_size, sample_size=args.sample_size)
            else:
                history = random_search(cycles=args.n_iters)

            fh = open(os.path.join(output_path,
                                   'algo_{}_{}_ssp_{}_seed_{}.obj'.format(alg,
                                                                          args.run_id,
                                                                          space,
                                                                          seed)), 'wb')
            pickle.dump(history, fh)
            fh.close()

            print(min([1 - arch.test_accuracy - search_space.test_min_error for arch in history]))
