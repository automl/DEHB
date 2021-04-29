import glob
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import rcParams
from matplotlib import pyplot as plt
rcParams['font.family'] = 'serif'


# Load run statistics to create (benchmark x algorithm) table showing mean final performance

list_of_mean_files = glob.glob('dehb/examples/*/results/mean_df.pkl') + \
                     glob.glob('dehb/examples/*/results/*/mean_df.pkl')

list_of_std_files = glob.glob('dehb/examples/*/results/std_df.pkl') + \
                    glob.glob('dehb/examples/*/results/*/std_df.pkl')
mean_dfs = {}
for filename in list_of_mean_files:
    benchname = filename.split('/')[-2]
    with open(filename, 'rb') as f:
        mean_dfs[benchname] = pickle.load(f)
mean_dfs = pd.DataFrame(mean_dfs).transpose()
mean_dfs.to_pickle("all_mean_dfs.pkl")

std_dfs = {}
for filename in list_of_std_files:
    benchname = filename.split('/')[-2]
    with open(filename, 'rb') as f:
        std_dfs[benchname] = pickle.load(f)
std_dfs = pd.DataFrame(std_dfs).transpose()
std_dfs.to_pickle("all_std_dfs.pkl")

# Load run statistics to create a relative ranking plot over time

rank_list_candidates = glob.glob('dehb/examples/*/results/rank_df.pkl') + \
                       glob.glob('dehb/examples/*/results/*/rank_df.pkl')
list_of_rank_files = []
for name in rank_list_candidates:
    # ignore benchmarks where the runtime is not wallclock time in seconds
    if "countingones" in name or "bnn" in name or "svm" in name:
        continue
    list_of_rank_files.append(name)


# load rankings per benchmark
rank_dfs = []
for filename in list_of_rank_files:
    with open(filename, 'rb') as f:
        rank_dfs.append(pickle.load(f))

# reorganize data to have algorithms as the top hierarchy, followed by every benchmark for the algo
avg_rank = {}
for i in range(len(rank_dfs)):
    for name in rank_dfs[i].columns:
        if name not in avg_rank.keys():
            avg_rank[name] = {}
        if i not in avg_rank[name].keys():
            avg_rank[name][i] = None
        avg_rank[name][i] = pd.Series(data=rank_dfs[i][name], index=rank_dfs[i].index)

# assigning mean rank to all algorithms at start
starting_rank = np.mean(np.arange(1, 1+len(avg_rank.keys())))
for name, v in avg_rank.items():
    avg_rank[name] = pd.DataFrame(v)
    avg_rank[name].iloc[0] = [starting_rank] * avg_rank[name].shape[1]

# compute mean relative rank of each algorithm across all benchmarks
rank_lists = {}
for name, v in avg_rank.items():
    rank_lists[name] = pd.Series(
        data=np.mean(avg_rank[name].ffill(), axis=1), index=avg_rank[name].index
    )
rank_lists = pd.DataFrame(rank_lists)

linestyles = [(0, (1, 5)),  # loosely dotted
              (0, (5, 5)),  # loosely dashed
              'dotted',
              (0, (3, 2, 1, 2, 1, 2)),  # dash dot dotted
              'dashed',
              'dashdot',
              (0, (3, 1, 1, 1, 1, 1)),
              'solid']

colors = ["C%d" % i for i in range(len(rank_lists.columns))]
if len(rank_lists.columns) <= 8:
    _colors = dict()
    _colors["RS"] = "C0"
    _colors["HB"] = "C7"
    _colors["BOHB"] = "C1"
    _colors["TPE"] = "C3"
    _colors["SMAC"] = "C4"
    _colors["RE"] = "C5"
    _colors["DE"] = "C6"
    _colors["DEHB"] = "C2"
    colors = []
    for l in rank_lists.columns:
        colors.append(_colors[l])


landmarks = np.arange(start=0, stop=rank_lists.shape[0], step=5)  # for smoothing
plt.clf()
xlims = [np.inf, -np.inf]
for i, name in enumerate(rank_lists.columns):
    if name == 'DEHB':
        lw, a = (1.75, 1)
    else:
        lw, a = (1.5, 0.7)
    plt.plot(
        rank_lists[name].index.to_numpy()[landmarks],
        rank_lists[name].to_numpy()[landmarks],
        label=name, alpha=a, linestyle=linestyles[i], linewidth=1.5
    )
    xlims[0] = min(xlims[0], rank_lists[name].index.to_numpy()[0])
    xlims[1] = max(xlims[1], rank_lists[name].index.to_numpy()[-1])

plt.xscale('log')
plt.legend(loc='upper left', framealpha=1, prop={'size': 12}, ncol=4)
plt.fill_between(
    rank_lists['DEHB'].index.to_numpy()[landmarks],
    0, rank_lists['DEHB'].to_numpy()[landmarks],
    alpha=0.5, color='gray'
)
# plt.fill_between(
#     rank_lists['DEHB'].index.to_numpy()[landmarks],
#     0, starting_rank,
#     alpha=0.3, color='gray'
# )
# plt.hlines(starting_rank, 0, 1e7)
plt.xlim(xlims[0], xlims[1])
plt.ylim(1, rank_lists.shape[1])
plt.xlabel('estimated wallclock time $[s]$', fontsize=15)
plt.ylabel('average relative rank', fontsize=15)
plt.savefig('rank_plot.pdf', bbox_inches='tight')

rank_stats = {}
rank_stats['minimum'] = np.min(rank_lists, axis=0)
rank_stats['maximum'] = np.max(rank_lists, axis=0)
rank_stats['variance'] = np.var(rank_lists, axis=0)
rank_stats = pd.DataFrame(rank_stats)


def row_entries_for(x):
    """ Function to get latex rows for table for the benchmark x passed as input
    """
    print(' $\pm$ & '.join(
        mean_dfs.loc[x].apply(np.format_float_scientific, precision=1, exp_digits=1).to_numpy())
    )
    print(' & '.join(
        std_dfs.loc[x].apply(np.format_float_scientific, precision=1, exp_digits=1).to_numpy())
    )
    model = mean_dfs.columns[np.argmin(mean_dfs.loc[x].to_list())]
    print(
        model,
        np.format_float_scientific(mean_dfs.loc[x][model], precision=1, exp_digits=1),
        np.format_float_scientific(std_dfs.loc[x][model], precision=1, exp_digits=1)
    )
    print(stats.rankdata(mean_dfs.loc[x]))
    print("DEHB's rank {}".format(stats.rankdata(mean_dfs.loc[x])[-1]))

# Ranks based on final numbers
rank_df = {}
for idx in mean_dfs.index:
    rank_df[idx] = pd.Series(data=stats.rankdata(mean_dfs.loc[idx]), index=mean_dfs.loc[idx].index)
rank_df = pd.DataFrame(rank_df)
print(rank_df.mean(axis=1))
