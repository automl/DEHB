import os
import json
import pickle
import collections
import numpy as np
import pandas as pd
from scipy import stats


def create_plot(plt, methods, path, regret_type, fill_trajectory,
                colors, linestyles, marker, n_runs=500, limit=1e7):

    # plot limits
    min_time = np.inf
    max_time = 0
    min_regret = 1
    max_regret = 0

    # stats for rank plot
    frame_dict = collections.OrderedDict()
    available_models = []

    dataset = path.replace('/', ' ').strip().split(' ')[-1]
    if dataset == 'cifar10-valid':
        y_star_valid, y_star_test = (0.08393333349609367, 0.08476666666666677)
    elif dataset == 'cifar100':
        y_star_valid, y_star_test = (0.26506666642252596, 0.2649666666666667)
    elif dataset == 'ImageNet16-120':
        y_star_valid, y_star_test = (0.5326666672770182, 0.531555555352105)
    else:
        raise "Cannot find dataset and global incumbent"

    # finding best found incumbent to be global incumbent
    min_max_time = []
    global_inc = np.inf
    for index, (m, label) in enumerate(methods):
        runtimes = []
        for k, i in enumerate(np.arange(n_runs)):
            try:
                res = json.load(open(os.path.join(path, m, "run_{}.json".format(i))))
            except Exception as e:
                print(m, i, e)
                runtimes.append(limit)
                continue
            regret_key =  "regret_validation" if regret_type == 'validation' else "regret_test"
            runtime_key = "runtime"
            curr_inc = np.min(np.array(res[regret_key]) + y_star_valid)
            if curr_inc < global_inc:
                global_inc = curr_inc
            runtimes.append(np.min((np.cumsum(res[runtime_key])[-1], limit)))
        min_max_time.append(np.mean(runtimes))
    limit = np.min((np.min(min_max_time), limit))
    print("Found global incumbent: ", global_inc, "\tMin-max time: ", limit)

    no_runs_found = False
    # looping and plotting for all methods
    for index, (m, label) in enumerate(methods):

        regret = []
        runtimes = []
        for k, i in enumerate(np.arange(n_runs)):
            try:
                res = json.load(open(os.path.join(path, m, "run_{}.json".format(i))))
                no_runs_found = False
            except Exception as e:
                print(m, i, e)
                no_runs_found = True
                continue
            regret_key = "regret_validation" if regret_type == 'validation' else "regret_test"
            runtime_key = "runtime"
            curr_regret = np.array(res[regret_key]) + y_star_valid - global_inc
            _, idx = np.unique(curr_regret, return_index=True)
            idx.sort()
            # regret.append(np.array(res[regret_key])[idx])
            regret.append(curr_regret[idx])
            runtimes.append(np.array(res[runtime_key])[idx])

        if not no_runs_found:
            # finds the latest time where the first measurement was made across runs
            t = np.max([runtimes[i][0] for i in range(len(runtimes))])
            min_time = min(min_time, t)
            te, time = fill_trajectory(regret, runtimes, replace_nan=1)

            idx = time.tolist().index(t)
            te = te[idx:, :]
            time = time[idx:]

            # Clips off all measurements after 10^7s
            idx = np.where(time < limit)[0]

            print("{}. Plotting for {}".format(index, m))
            print(len(regret), len(runtimes), len(idx))
            print("\nMean: {}; Std: {}\n".format(np.mean(te, axis=1)[idx][-1],
                                                 stats.sem(te[idx], axis=1)[-1]))

            # stats for rank plot
            frame_dict[str(m)] = pd.Series(data=np.mean(te, axis=1)[idx], index=time[idx])

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

    rank_stats = pd.DataFrame(frame_dict)
    rank_stats = rank_stats.ffill()

    # dividing log-scale range of [0, 1e6] into 1000 even buckets
    buckets = 50
    max_limit = 7.0
    t = 10 ** np.arange(start=0, stop=max_limit, step=max_limit/buckets)
    # creating dummy filler data to create the data frame
    d = np.random.uniform(size=(len(t), rank_stats.shape[-1]))
    d.fill(np.nan)
    # getting complete time range
    index = np.concatenate((t, rank_stats.index.to_numpy()))
    # concatenating actual and dummy data
    data = np.vstack((d, rank_stats))
    # ordering time
    idx = np.argsort(index)
    # creating new ordered data frame
    rank_stats = pd.DataFrame(data=data[idx], index=index[idx])
    rank_stats = rank_stats.ffill().loc[t]
    # replacing scores with the relative ranks
    rank_stats = rank_stats.apply(np.argsort, axis=1)
    # to start ranks from '1'
    rank_stats += 1
    # assigning an equal average rank to all agorithms at the beginning
    rank_stats = rank_stats.replace(0, np.mean(np.arange(rank_stats.shape[-1]) + 1))
    # adding model/column names
    rank_stats.columns = available_models

    dataset = path.replace('/', ' ').strip().split(' ')[-1]
    with open('{}.pkl'.format(dataset), 'wb') as f:
        pickle.dump(rank_stats, f)

    return plt, min_time, max_time, min_regret, max_regret
