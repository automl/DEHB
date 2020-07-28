# Slightly modified version of:
# https://github.com/automl/nas_benchmarks/blob/development/experiment_scripts/run_regularized_evolution.py

"""
Regularized evolution as described in:
Real, E., Aggarwal, A., Huang, Y., and Le, Q. V.
Regularized Evolution for Image Classifier Architecture Search.
In Proceedings of the Conference on Artificial Intelligence (AAAI’19)

The code is based one the original regularized evolution open-source implementation:
https://colab.research.google.com/github/google-research/google-research/blob/master/evolution/regularized_evolution_algorithm/regularized_evolution.ipynb

"""

import os
import sys
import json
import random
import argparse
import collections
import ConfigSpace
import numpy as np
from copy import deepcopy

from hpolib.benchmarks.ml.bnn_benchmark import BNNOnBostonHousing, BNNOnProteinStructure
from hpolib.benchmarks.ml.bnn_benchmark import BNNOnToyFunction, BNNOnYearPrediction


global_cost = []

class Model(object):
    """A class representing a model.

    It holds two attributes: `arch` (the simulated architecture) and `accuracy`
    (the simulated accuracy / fitness). See Appendix C for an introduction to
    this toy problem.

    In the real case of neural networks, `arch` would instead hold the
    architecture of the normal and reduction cells of a neural network and
    accuracy would be instead the result of training the neural net and
    evaluating it on the validation set.

    We do not include test accuracies here as they are not used by the algorithm
    in any way. In the case of real neural networks, the test accuracy is only
    used for the purpose of reporting / plotting final results.

    In the context of evolutionary algorithms, a model is often referred to as
    an "individual".

    Attributes:
      arch: the architecture as an int representing a bit-string of length `DIM`.
          As a result, the integers are required to be less than `2**DIM`. They
          can be visualized as strings of 0s and 1s by calling `print(model)`,
          where `model` is an instance of this class.
      accuracy:  the simulated validation accuracy. This is the sum of the
          bits in the bit-string, divided by DIM to produce a value in the
          interval [0.0, 1.0]. After that, a small amount of Gaussian noise is
          added with mean 0.0 and standard deviation `NOISE_STDEV`. The resulting
          number is clipped to within [0.0, 1.0] to produce the final validation
          accuracy of the model. A given model will have a fixed validation
          accuracy but two models that have the same architecture will generally
          have different validation accuracies due to this noise. In the context
          of evolutionary algorithms, this is often known as the "fitness".
    """

    def __init__(self):
        self.arch = None
        self.accuracy = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return '{0:b}'.format(self.arch)


def train_and_eval(config):
    global max_budget, b
    res = b.objective_function(config, budget=int(max_budget))
    fitness = res['function_value']
    cost = max_budget
    global_cost.append(cost)
    return -fitness    # RE maximizes
    # y, cost = b.objective_function(config)
    # global_cost.append(cost)
    # return -y


def random_architecture():
    config = cs.sample_configuration()
    return config


def mutate_arch(parent_arch):
    # pick random parameter
    dim = np.random.randint(len(cs.get_hyperparameters()))
    hyper = cs.get_hyperparameters()[dim]

    if type(hyper) == ConfigSpace.UniformFloatHyperparameter:
        while True:
            value = np.random.uniform(hyper.lower, hyper.upper)
            if parent_arch[hyper.name] != value:
                child_arch = deepcopy(parent_arch)
                child_arch[hyper.name] = value
                return child_arch
    else:
        if type(hyper) == ConfigSpace.OrdinalHyperparameter:
            choices = list(hyper.sequence)
        elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
            choices = list(hyper.choices)
        elif type(hyper) == ConfigSpace.UniformIntegerHyperparameter:
            choices = np.arange(hyper.lower, hyper.upper + 1).tolist()
        # drop current values from potential choices
        choices.remove(parent_arch[hyper.name])

    # flip parameter
    idx = np.random.randint(len(choices))

    child_arch = deepcopy(parent_arch)
    child_arch[hyper.name] = choices[idx]
    return child_arch


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
        model.accuracy = train_and_eval(model.arch)
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
        parent = max(sample, key=lambda i: i.accuracy)

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch)
        # print(child.arch.get_array())
        child.accuracy = train_and_eval(child.arch)
        population.append(child)
        history.append(child)

        # Remove the oldest model.
        population.popleft()

    return history


def convert_to_json(history):
    global global_cost
    regret_validation = []
    regret_test = []
    runtime = []
    inc = np.inf
    test_regret = 1
    validation_regret = 1

    for i in range(len(history)):
        architecture = history[i]
        validation_regret = -architecture.accuracy
        if validation_regret <= inc:
            inc = validation_regret
        regret_validation.append(inc)
        regret_test.append(inc)
        # runtime.append(cost)

    res = {}
    res['regret_validation'] = regret_validation
    res['regret_test'] = regret_test
    res['runtime'] = np.cumsum(global_cost).tolist()
    return res


parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--dataset', help="name of the dataset used", default='bostonhousing',
                    choices=['toyfunction', 'bostonhousing', 'proteinstructure', 'yearprediction'])
parser.add_argument('--n_iters', default=10, type=int, nargs='?',
                    help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--pop_size', default=100, type=int, nargs='?', help='population size')
parser.add_argument('--sample_size', default=10, type=int, nargs='?', help='sample_size')
parser.add_argument('--max_budget', default=10000, type=int, nargs='?', help='minimum budget')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--folder', default="regularized_evolution", type=str, nargs='?',
                    help='folder where output is dumped')

args = parser.parse_args()
print(args)
max_budget = args.max_budget

if args.dataset == 'bostonhousing':
    b = BNNOnBostonHousing()
elif args.dataset == 'proteinstructure':
    b = BNNOnProteinStructure()
elif args.dataset == 'toyfunction':
    b = BNNOnToyFunction()
else:
    b = BNNOnYearPrediction()

cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

output_path = os.path.join(args.output_path, args.dataset, args.folder)
os.makedirs(os.path.join(output_path), exist_ok=True)

runs = args.runs

if runs is None:
    history = regularized_evolution(
        cycles=args.n_iters, population_size=args.pop_size, sample_size=args.sample_size)

    res = convert_to_json(history)

    fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
    json.dump(res, fh)
    fh.close()
else:
    ### Multiple runs
    for run_id in range(runs):

        print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        history = regularized_evolution(
            cycles=args.n_iters, population_size=args.pop_size, sample_size=args.sample_size)

        res = convert_to_json(history)

        fh = open(os.path.join(output_path, 'run_%d.json' % run_id), 'w')
        json.dump(res, fh)
        fh.close()
        print("Run saved. Resetting...")
