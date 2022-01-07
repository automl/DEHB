import scipy.stats as sps
import matplotlib.pyplot as plt
import pickle
import numpy as np


def load_pickle(filename):

    with open(filename, "rb") as f:
        var = pickle.load(f)

    return var


def correlation_across_budgets(filename=None, show=False):
    '''
    Based on code from here: 
    https://automl.github.io/HpBandSter/build/html/_modules/hpbandster/visualization.html#correlation_across_budgets


    Parameters
    ----------
    filename : str
        If filename is None, it uses the latest modified .pkl file in the sub-directory temp of the current working directory
    show : bool
        Whether to show resulting plot or not

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot)
    
    
    '''

    if filename is None:
        import os
        os.chdir('temp')
        files = os.listdir()
        files.sort(key=os.path.getmtime)
        for file in files[::-1]:
            if file[-4:] == '.pkl':
                filename = file[:-4]
                break
        os.chdir('..')

        print('No filename provided. Loading the latest modified .pkl file in sub-directory '
                'temp with name: ' + filename + '.pkl.')

    complete_trajectory = load_pickle("./temp/" + filename + ".pkl")
    print("Loaded trajectory from pickle:", complete_trajectory)

    stats_dict = {}
    for r in complete_trajectory:
        # Check if config is already in our stats dict
        if str(r[0]) not in stats_dict:
            stats_dict[str(r[0])] = []

        stats_dict[str(r[0])].append({"fitness": r[1], "cost": r[2], "budget": r[3]})

    
    budgets = list(set([r[3] for r in complete_trajectory]))
    budgets.sort()

    import itertools

    loss_pairs = {}
    for b in budgets[:-1]:
        loss_pairs[b] = {}

    for b1, b2 in itertools.combinations(budgets, 2):
        loss_pairs[b1][b2]= []
    print("loss_pairs:", loss_pairs)

    for config in stats_dict:
        print(stats_dict[config])
        if len(stats_dict[config]) < 2: continue
        
        for r1, r2 in itertools.combinations(stats_dict[config], 2):
            if not np.isfinite(r1["fitness"]) or not np.isfinite(r2["fitness"]): continue
            loss_pairs[float(r1["budget"])][float(r2["budget"])].append((r1["fitness"], r2["fitness"]))

    print("budgets:", budgets)
    print("loss_pairs:", loss_pairs)


    rhos = np.eye(len(budgets)-1)
    rhos.fill(np.nan)

    ps = np.eye(len(budgets)-1)
    ps.fill(np.nan)

    for i in range(len(budgets)-1):
        for j in range(i+1,len(budgets)):
            spr = sps.spearmanr(loss_pairs[budgets[i]][budgets[j]])
            rhos[i][j-1] = spr.correlation
            ps[i][j-1] = spr.pvalue


    fig, ax = plt.subplots()

    cax = ax.matshow(rhos, vmin=-1, vmax=1)
    fig.colorbar(cax)


    ax.set_yticks( range(len(budgets)-1))
    ax.set_yticklabels(budgets[:-1],)

    ax.set_xticks( range(len(budgets)-1))
    ax.set_xticklabels(budgets[1:],)
    
    ax.set_title('Rank correlation of the loss across the budgets')

    for i in range(len(budgets)-1):
        for j in range(i+1,len(budgets)):
            plt.text(j-1,i, r'$\rho_{spearman}= %f$'%rhos[i][j-1] + '\n' + r'$p = %f$'%ps[i][j-1] + 
                        '\n' + r'$n = %i$'%len(loss_pairs[budgets[i]][budgets[j]]),
                        horizontalalignment='center', verticalalignment='center')


    if show:
        plt.show()

    print(type(fig), type(ax))
    return(fig, ax)

