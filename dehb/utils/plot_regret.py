'''Script to plot regret curves for multiple runs on the benchmarks
'''

import os
import json
import sys
import pickle
import argparse
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn
#seaborn.set_style("ticks")

from matplotlib import rcParams
rcParams["font.size"] = "30"
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['figure.figsize'] = (16.0, 9.0)
rcParams['figure.frameon'] = True
rcParams['figure.edgecolor'] = 'k'
rcParams['grid.color'] = 'k'
rcParams['grid.linestyle'] = ':'
rcParams['grid.linewidth'] = 0.5
rcParams['axes.linewidth'] = 1
rcParams['axes.edgecolor'] = 'k'
rcParams['axes.grid.which'] = 'both'
rcParams['legend.frameon'] = 'True'
rcParams['legend.framealpha'] = 1

rcParams['ytick.major.size'] = 12
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.size'] = 6
rcParams['ytick.minor.width'] = 1
rcParams['xtick.major.size'] = 12
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.size'] = 6
rcParams['xtick.minor.width'] = 1

marker=['x', '^', 'D', 'o', 's', 'h', '*', 'v', '<', ">"]
linestyles = ['-', '--', '-.', ':']


def fill_trajectory(performance_list, time_list, replace_nan=np.NaN):
    frame_dict = collections.OrderedDict()
    counter = np.arange(0, len(performance_list))
    for p, t, c in zip(performance_list, time_list, counter):
        if len(p) != len(t):
            raise ValueError("(%d) Array length mismatch: %d != %d" %
                             (c, len(p), len(t)))
        frame_dict[str(c)] = pd.Series(data=p, index=t)

    # creates a dataframe where the rows are indexed based on time
    # fills with NA for missing values for the respective timesteps
    merged = pd.DataFrame(frame_dict)
    # ffill() acts like a fillna() wherein a forward fill happens
    # only remaining NAs for in the beginning until a value is recorded
    merged = merged.ffill()

    performance = merged.get_values()  # converts to a 2D numpy array
    time_ = merged.index.values        # retrieves the timestamps

    performance[np.isnan(performance)] = replace_nan

    if not np.isfinite(performance).all():
        raise ValueError("\nCould not merge lists, because \n"
                         "\t(a) one list is empty?\n"
                         "\t(b) the lists do not start with the same times and"
                         " replace_nan is not set?\n"
                         "\t(c) any other reason.")

    return performance, time_


parser = argparse.ArgumentParser()

parser.add_argument('--benchmark', default='101', type=str, nargs='?',
                    choices=['101', '1shot1', '201', 'paramnet', 'svm',
                             'countingones', 'rl', 'bnn', 'cc18'],
                    help='select benchmark to plot')
parser.add_argument('--bench_type', default='protein', type=str, nargs='?',
                    help='select subset of benchmark to plot')
parser.add_argument('--path', default='./', type=str, nargs='?',
                    help='path to encodings or jsons for each algorithm')
parser.add_argument('--file', default='./', type=str, nargs='?',
                    help='path and filename to list of algorithms to plot')
parser.add_argument('--n_runs', default=10, type=int, nargs='?',
                    help='number of runs to plot data for')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the plot will be saved')
parser.add_argument('--type', default="wallclock", type=str, choices=["wallclock", "fevals"],
                    help='to plot for wallclock times or # function evaluations')
parser.add_argument('--name', default="comparison", type=str,
                    help='file name for the PNG plot to be saved')
parser.add_argument('--title', default="benchmark", type=str,
                    help='title name for the plot')
parser.add_argument('--limit', default=1e7, type=float, help='wallclock limit')
parser.add_argument('--regret', default='test', type=str, choices=['validation', 'test'],
                    help='type of regret')

args = parser.parse_args()
path = args.path
n_runs = args.n_runs
limit = args.limit
plot_type = args.type
plot_name = args.name
regret_type = args.regret
benchmark = args.benchmark
bench_type = args.bench_type

# Checking benchmark specifications
if benchmark == '101':
    from dehb.examples.nas101 import create_plot

elif benchmark == '1shot1' and bench_type not in ["1", "2", "3"]:
    print("Specify \'--bench_type\' from {1, 2, 3} for choosing the search space for 1shot1.")
    sys.exit()
elif benchmark == '1shot1' :
    ssp = bench_type
    from dehb.examples.nas1shot1 import create_plot

elif benchmark == 'countingones':
    from dehb.examples.countingones import create_plot

elif benchmark == 'paramnet':
    from dehb.examples.paramnet import create_plot

elif benchmark == 'svm':
    from dehb.examples.svm import create_plot

elif benchmark == '201':
    from dehb.examples.nas201 import create_plot

elif benchmark == 'rl':
    from dehb.examples.cartpole import create_plot

elif benchmark == 'bnn':
    from dehb.examples.bnn import create_plot

elif benchmark == 'cc18':
    from dehb.examples.cc18 import create_plot

# Loading file for algo list
with open(args.file, 'r') as f:
    methods = eval(f.readlines()[0])


# plot limits
min_time = np.inf
max_time = 0
min_regret = 1
max_regret = 0

# plot setup
colors = ["C%d" % i for i in range(len(methods))]
plt.clf()

no_runs_found = False

if benchmark == '1shot1':
    plt, min_time, max_time, min_regret, max_regret = \
        create_plot(plt, methods, path, regret_type, fill_trajectory,
                    colors, linestyles, marker, n_runs, limit, ssp)
else:
    plt, min_time, max_time, min_regret, max_regret = \
        create_plot(plt, methods, path, regret_type, fill_trajectory,
                    colors, linestyles, marker, n_runs, limit)


if True: #benchmark != 'cc18':
    plt.xscale("log")
if benchmark != 'svm' and benchmark != 'bnn':
     plt.yscale("log")
plt.tick_params(which='both', direction="in")
if benchmark == 'svm':
    plt.legend(loc='upper right', framealpha=1, prop={'size': 30, 'weight': 'bold'})
else:
    plt.legend(loc='lower left', framealpha=1, prop={'size': 30, 'weight': 'bold'})
plt.title(args.title)

if benchmark == 'rl':
    plt.xlabel("time $[s]$", fontsize=50)
elif benchmark == 'bnn':
    plt.xlabel("MCMC steps", fontsize=50)
elif benchmark == 'countingones':
    plt.xlabel("cummulative budget / $b_{max}$", fontsize=50)
elif plot_type == "wallclock":
    plt.xlabel("estimated wallclock time $[s]$", fontsize=50)
elif plot_type == "fevals":
    plt.xlabel("number of function evaluations", fontsize=50)

if benchmark == 'svm':
    plt.ylabel("{} error".format(regret_type), fontsize=50)
elif benchmark == 'rl':
    plt.ylabel("epochs until convergence", fontsize=50)
elif benchmark == 'bnn':
    plt.ylabel("negative log-likelihood", fontsize=50)
elif benchmark == 'countingones':
    plt.ylabel("normalized {} regret".format(regret_type))
else:
    plt.ylabel("{} regret".format(regret_type), fontsize=50)

if benchmark == 'rl':
    plt.xlim(1e1, 1e5)
elif benchmark == 'bnn':
    plt.xlim(1e4, 1e6)
elif benchmark == 'countingones':
    plt.xlim(max(min_time/10, 1e-1), min(max_time*10, 1e7))
elif benchmark == 'cc18':
    plt.xlim(0, max_time)
else:
    plt.xlim(max(min_time/10, 1e0), min(max_time*10, 1e7))

if benchmark == 'bnn':
    plt.ylim(3, 75)
elif benchmark == 'rl':
    plt.ylim(1e2, 1e4)
else:
    plt.ylim(min_regret, max_regret)

plt.grid(which='both', alpha=0.5, linewidth=0.5)
print(os.path.join(args.output_path, '{}.png'.format(plot_name)))
plt.savefig(os.path.join(args.output_path, '{}.png'.format(plot_name)),
            bbox_inches='tight', dpi=300)
