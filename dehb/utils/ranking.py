
from matplotlib import rcParams
rcParams['font.family'] = 'serif'


linestyles = [(0, (1, 5)),  # loosely dotted
              (0, (5, 5)),  # loosely dashed
              'dotted',
              (0, (3, 2, 1, 2, 1, 2)),  # dash dot dotted
              'dashed', 'dashdot', 'solid']

plt.clf()
for i in range(7):
    if i == 6:
        lw, a = (1.5, 1)
    elif i == 5:
        lw, a = (1.5, 1)
    else:
        lw, a = (1.5, 0.8)
    plt.plot(protein.index.to_list(), final_ranks[:, i], label=names[i],
             linestyle=linestyles[i], linewidth=lw, alpha=a)

plt.xscale('log');
plt.legend(loc='upper left', framealpha=1, prop={'size': 12}, ncol=4)
plt.fill_between(protein.index.to_list(), 0, final_ranks[:, -1], alpha=0.3, color='pink')
plt.xlim(1e1, 1e6)
plt.xlabel('estimated wallclock time $[s]$', fontsize=15)
plt.ylabel('average relative rank', fontsize=15)
plt.tight_layout()
