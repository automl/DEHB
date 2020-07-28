def prep(traj, runtime, history):
    global X, budgets, fitness, fidelities, configs
    budgets = [elem[-1]  for elem in history]
    fitness = [elem[1]  for elem in history]
    configs = np.array([elem[0]  for elem in history])
    X = get_mds(configs)
    fidelities = None
    fidelities = {}
    for budget in np.unique(budgets):
        fidelities[budget] = np.where(budgets == budget)[0]

    xlim = (np.min(X[:,0])-0.5, np.max(X[:,0])+0.5)
    ylim = (np.min(X[:,1])-0.5, np.max(X[:,1])+0.5)

    b_size = len(np.unique(budgets))
    plt.clf()
    # fig, ax = plt.subplots(1, b_size, sharey=True, sharex=True)
    fig, ax = plt.subplots(2, 2, sharey=True, sharex=True)
    incs = np.where(traj == fitness)[0]
    rgba_colors = generate_colors(X.shape[0], budgets)
    gridify = {np.unique(budgets)[0]: [0, 0], np.unique(budgets)[1]: [0, 1],
               np.unique(budgets)[2]: [1, 0], np.unique(budgets)[3]: [1, 1]}
    for i, key in enumerate(fidelities):
        x = fidelities[key]
        m, n = gridify[key]
        ax[m, n].scatter(X[x, 0][::-1], X[x, 1][::-1], color=rgba_colors[x][::-1], label=key, s=20)
        idxs = np.intersect1d(x, incs, assume_unique=True)
        if len(idxs) > 0:
            rgba_colors[idxs, :3] = 0
            ax[m, n].scatter(X[idxs, 0][::-1], X[idxs, 1][::-1], color=rgba_colors[idxs][::-1],
                          label='incumbent', marker='v', s=70)
        ax[m, n].legend(prop={'size': 7})
    plt.suptitle('Trajectory for different budgets', fontsize=20)



def expanded_dimensions(cs):
    dims = 0
    new_dim_idx = 0
    dim_map = {}
    for i, hyper in enumerate(cs.get_hyperparameters()):
        if isinstance(hyper, ConfigSpace.CategoricalHyperparameter):
            dims += len(hyper.choices)
            dim_map[i] = np.arange(start=new_dim_idx, stop=new_dim_idx + len(hyper.choices), step=1)
            new_dim_idx += len(hyper.choices)
        elif isinstance(hyper, ConfigSpace.CategoricalHyperparameter):
            dims += len(hyper.sequence)
            dim_map[i] = np.arange(start=new_dim_idx, stop=new_dim_idx + len(hyper.sequence), step=1)
            new_dim_idx += len(hyper.sequence)
        else:
            dims += 1
            dim_map[i] = [new_dim_idx]
            new_dim_idx += 1
    return dims, dim_map

def map_to_original(vector, dim_map):
    dimensions = len(dim_map.keys())
    new_vector = np.random.uniform(size=dimensions)
    for i in range(dimensions):
        new_vector[i] = np.max(np.array(vector)[dim_map[i]])
    return new_vector
