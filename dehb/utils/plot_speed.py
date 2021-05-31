from matplotlib import rcParams
rcParams["font.size"] = "25"
rcParams['text.usetex'] = False
rcParams['font.family'] = 'serif'
rcParams['figure.figsize'] = (16.0, 9.0)
rcParams['figure.frameon'] = True
rcParams['legend.frameon'] = 'True'
rcParams['legend.framealpha'] = 1

dehb['runtime'] = np.array(dehb['runtime']) - dehb['runtime'][0]
bohb['runtime'] = np.array(bohb['runtime']) - bohb['runtime'][0]

plt.plot(bohb['runtime'], np.arange(start=1, stop=len(bohb['regret_validation'])+1, step=1), label='BOHB', color='orange', linewidth=5)
plt.plot(dehb['runtime'], np.arange(start=1, stop=len(dehb['regret_validation'])+1, step=1), label='DEHB', color='green', linewidth=5)

plt.title("Speed comparison BOHB vs DEHB", fontsize=45)
plt.legend(loc='upper left', framealpha=1, prop={'size': 45, 'weight': 'normal'})
plt.xscale('log'); plt.yscale('log')
plt.xlabel("Wallclock time for only optimization $(s)$", fontsize=45)
plt.ylabel("# function evaluations", fontsize=45)
# plt.ylim(2, len(bohb['regret_validation']))

plt.savefig('temp/cifar10_speed.png', dpi=300, bbox_inches="tight")

# plt.savefig('temp/mnist_speed.png', dpi=300)
# plt.show()
