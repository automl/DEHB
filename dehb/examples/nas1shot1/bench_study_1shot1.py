import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nasbench/'))
sys.path.append(os.path.join(os.getcwd(), '../nasbench-1shot1/'))

import argparse
import numpy as np
import pandas as pd
from matplotlib import cm as CM
from matplotlib import pyplot as plt
from scipy.stats import spearmanr as corr

from nasbench import api

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3

import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'dehb/utils'))
from plot_mds import vector_to_configspace, get_mds


################
# Common Utils #
################

def final_score_relation(sample_size=1e6, output=None):
    global b, cs, nasbench
    x = []
    y = []
    b.run_history = []
    for i in range(int(sample_size)):
        print("{:<6}/{:<6}".format(i+1, sample_size), end='\r')
        config = cs.sample_configuration()
        valid_score, _ = b.objective_function(nasbench, config)
        test_score = b.run_history[i].test_accuracy
        x.append(valid_score)
        y.append(test_score)
    xlim = (min(x), max(x))
    ylim = (min(y), max(y))
    plt.clf()
    plt.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], color='black',
             alpha=0.5, linestyle='--', linewidth=1)
    plt.scatter(x, y)
    plt.xlabel('Validation scores')
    plt.ylabel('Test scores')
    if output is None:
        plt.show()
    else:
        plt.savefig(output, dpi=300)


def budget_correlation(sample_size, budgets, compare=False, output=None):
    global b, cs
    df = pd.DataFrame(columns=budgets, index=np.arange(sample_size))
    for i in range(int(sample_size)):
        print("{:<6}/{:<6}".format(i+1, sample_size), end='\r')
        config = cs.sample_configuration()
        for j, budget in enumerate(budgets):
            score, _ = b.objective_function(nasbench, config, budget=budget)
            df.iloc[i, j] = score
        b.run_history = []
    res = corr(df)
    corr_val = res.correlation

    plt.clf()
    ax = plt.gca()
    mat = ax.matshow(corr_val)
    for i in range(len(corr_val)):
        for j in range(len(corr_val[0])):
            ax.text(j, i, "{:0.5f}".format(corr_val[i][j]), ha="center", va="center", rotation=45)
    # Major ticks
    ax.set_xticks(np.arange(0, len(budgets), 1))
    ax.set_yticks(np.arange(0, len(budgets), 1))
    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, len(budgets)+1, 1))
    ax.set_yticklabels(np.arange(1, len(budgets)+1, 1))
    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(budgets), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(budgets), 1), minor=True)
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    if compare:
        mat.set_clim(0, 1)
    else:
        mat.set_clim(np.min(corr_val), np.max(corr_val))
    plt.colorbar(mat)
    if output is None:
        plt.show()
    else:
        plt.savefig(output, dpi=300)


parser = argparse.ArgumentParser()
parser.add_argument('--space', default=1, type=int, nargs='?', choices=[1, 2, 3],
                    help='Search Space #')
parser.add_argument('--sample_size', default=1e4, type=float, nargs='?',
                    help='# samples')
parser.add_argument('--compare', default='False', type=str, nargs='?',
                    help='If True, color limits set to [-1, 1], else dynamic')
args = parser.parse_args()
space = args.space
sample_size = args.sample_size
compare = True if args.compare == 'True' else False


def plot_budget_landscape(budgets, sample_size=1000, output=None):
    print("Initialising...")
    x = np.random.uniform(size=(sample_size, dimensions))
    print("MDS conversion...")
    X = get_mds(x)
    print("Calculating budget scores...")
    scores = {}
    for budget in budgets:
        print("For budget {}".format(budget))
        scores[budget] = []

        for i in range(x.shape[0]):
            print("{:<4}/{:<4}".format(i + 1, x.shape[0]), end='\r')
            score, _ = f(config=vector_to_configspace(cs, x[i]), budget=budget)
            # score is error in [0, 1]
            scores[budget].append(1 - score)   # accuracy

    print("Plotting...")
    col = CM.plasma
    fig, axes = plt.subplots(np.ceil(len(budgets) / 2).astype(int), 2)
    for i, ax in enumerate(axes.flat):
        if i == len(budgets):
            break
        im = ax.hexbin(X[:,0], X[:,1], C=scores[budgets[i]], gridsize=30, cmap=col)
        ax.set_title(budgets[i])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax)

    plt.suptitle(name)
    if len(budgets) % 2 != 0:
        fig.delaxes(axes[np.floor(len(budgets) / 2).astype(int), 1])

    if output is None:
        plt.show()
    else:
        plt.savefig(output, dpi=300)


def f(config, budget=None):
    if budget is not None:
        fitness, cost = b.objective_function(nasbench, config, budget=int(budget))
    else:
        fitness, cost = b.objective_function(nasbench, config)
    fitness = 1 - fitness
    return fitness, cost


####################
# NAS-Bench-1shot1 #
####################

data_dir = "../nasbench-1shot1/nasbench_analysis/nasbench_data/108_e/nasbench_full.tfrecord"
nasbench = api.NASBench(data_dir)
budgets = [4, 12, 36, 108]

b = eval('SearchSpace{}()'.format(space))
cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

name = 'ss{}'.format(space)

plot_budget_landscape(budgets, sample_size=sample_size,
                      output='dehb/examples/plots/landscape/{}.png'.format(name))
# final_score_relation(sample_size,
#                      output='dehb/examples/plots/correlation/{}_test_val.png'.format(name))
# budget_correlation(sample_size, budgets=budgets, compare=compare,
#                    output='dehb/examples/plots/correlation/{}_{}.png'.format(name, compare))
