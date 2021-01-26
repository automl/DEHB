import os
import json
import pickle
import collections
import numpy as np
import pandas as pd
from scipy import stats


def create_plot(plt, methods, path, regret_type, fill_trajectory,
                colors, linestyles, marker, n_runs=500, limit=1e7, **kwargs):

    # plot limits
    min_time = 0  #np.inf
    max_time = 0
    min_regret = 1
    max_regret = 0

    # stats for rank plot
    frame_dict = collections.OrderedDict()
    available_models = []

    # for table and ranking plot
    mean_df = {}
    std_df = {}

    no_runs_found = False
    # looping and plotting for all methods
    for index, (m, label) in enumerate(methods):

        regret = []
        runtimes = []
        for k, i in enumerate(np.arange(n_runs)):
            try:
                if 'de' in m or 'evolution' in m:
                    res = json.load(open(os.path.join(path, m, "run_{}.json".format(i))))
                else:
                    res = pickle.load(open(os.path.join(path, m,
                                                        "{}_run_{}.pkl".format(m, i)), 'rb'))
                no_runs_found = False
            except Exception as e:
                print(m, i, e)
                no_runs_found = True
                continue
            if 'de' in m or 'evolution' in m:
                regret_key =  "regret_validation" if regret_type == 'validation' else "regret_test"
                runtime_key = "runtime"
            else:
                regret_key =  "losses" if regret_type == 'validation' else "test_losses"
                runtime_key = "cummulative_cost"
            curr_regret = np.array(res[regret_key])
            _, idx = np.unique(curr_regret, return_index=True)
            idx.sort()
            regret.append(curr_regret[idx])
            runtimes.append(np.array(res[runtime_key])[idx])

        if not no_runs_found:
            # finds the latest time where the first measurement was made across runs
            ## unlike other benchmarks, this benchmark has stochastic and noisy signals
            ## taking the minimum of the first recorded measurements across runs as starting point
            t = np.min([runtimes[i][0] for i in range(len(runtimes))])
            min_time = max(min_time, t)
            te, time = fill_trajectory(regret, runtimes, replace_nan=3000)

            idx = time.tolist().index(t)
            te = te[idx:, :]
            time = time[idx:]

            # Clips off all measurements after 10^7s
            idx = np.where(time <= limit)[0]

            print("{}. Plotting for {}".format(index, m))
            print(len(regret), len(runtimes))
            print("\nMean: {}; Std: {}\n".format(np.mean(te, axis=1)[idx][-1],
                                                 stats.sem(te[idx], axis=1)[-1]))

            # The mean plot
            plt.plot(time[idx], np.mean(te, axis=1)[idx], color=colors[index],
                     linewidth=4, label=label, linestyle=linestyles[index % len(linestyles)],
                     marker=marker[index % len(marker)], markevery=(0.1,0.1), markersize=15)
            # The error band
            plt.fill_between(time[idx],
                             np.mean(te, axis=1)[idx] + 2 * stats.sem(te[idx], axis=1),
                             np.mean(te[idx], axis=1)[idx] - 2 * stats.sem(te[idx], axis=1),
                             color="C%d" % index, alpha=0.2)

            available_models.append(label)
            # Stats to dynamically impose limits on the axes of the plots
            max_time = max(max_time, time[idx][-1])
            min_regret = min(min_regret, np.mean(te, axis=1)[idx][-1])
            max_regret = max(max_regret, np.mean(te, axis=1)[idx][0])

            # For final score table
            mean_df[label] = pd.Series(data=np.mean(te, axis=1)[idx], index=time[idx])
            std_df[label] = pd.Series(data=np.std(te, axis=1)[idx], index=time[idx])

    mean_df = pd.DataFrame(mean_df)
    std_df = pd.DataFrame(std_df)
    # minimum of the maximum time limit recorded for each algorithm
    cutoff_idx = min(
        list(map(lambda x: np.where(~mean_df.isna()[x] == True)[0][-1], mean_df.columns))
    )
    mean_df = mean_df.iloc[:cutoff_idx + 1].ffill()
    std_df = std_df.iloc[:cutoff_idx + 1].ffill()
    rank_df = mean_df.apply(stats.rankdata, axis=1, result_type='broadcast')
    mean_df.to_pickle(os.path.join(path, 'all_mean_df.pkl'))
    mean_df.iloc[-1].to_pickle(os.path.join(path, 'mean_df.pkl'))
    std_df.iloc[-1].to_pickle(os.path.join(path, 'std_df.pkl'))
    rank_df.to_pickle(os.path.join(path, 'rank_df.pkl'))

    return plt, min_time, max_time, min_regret, max_regret
