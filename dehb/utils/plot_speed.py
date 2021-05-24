dehb['runtime'] = np.array(dehb['runtime']) - dehb['runtime'][0]
bohb['runtime'] = np.array(bohb['runtime']) - bohb['runtime'][0]

plt.plot(bohb['runtime'], np.arange(start=1, stop=len(bohb['regret_validation'])+1, step=1), label='BOHB', color='#90CE73')
plt.plot(dehb['runtime'], np.arange(start=1, stop=len(dehb['validation_score'])+1, step=1), label='DEHB', color='#9B8CD4')

plt.title("Speed comparison BOHB vs DEHB", fontsize=22)
plt.legend(loc='upper left', framealpha=1, prop={'size': 25, 'weight': 'normal'})
plt.xscale('log'); plt.yscale('log')
plt.xlabel("Wallclock time for only optimization $(s)$", fontsize=22)
plt.ylabel("# function evaluations", fontsize=22)
# plt.ylim(2, len(bohb['regret_validation']))

plt.savefig('temp/cifar10_speed.png', dpi=300)

# plt.savefig('temp/mnist_speed.png', dpi=300)
# plt.show()
