dehb['runtime'] = np.array(dehb['runtime']) - dehb['runtime'][0]
bohb['runtime'] = np.array(bohb['runtime']) - bohb['runtime'][0]

plt.plot(bohb['runtime'], np.arange(start=1, stop=len(bohb['regret_validation'])+1, step=1), label='BOHB', color='#90CE73')
plt.plot(dehb['runtime'], np.arange(start=1, stop=len(dehb['validation_score'])+1, step=1), label='DEHB', color='#9B8CD4')

plt.title("Speed comparison BOHB vs DEHB", )
plt.legend(loc='upper left', framealpha=1, prop={'size': 100, 'weight': 'bold'})
plt.xscale('log'); plt.yscale('log'); plt.legend()
plt.xlabel("Wallclock time sans function evaluation time $(s)$", fontsize=12)
plt.ylabel("Number of function evaluations", fontsize=12)
# plt.ylim(2, len(bohb['regret_validation']))

plt.savefig('temp/cifar10_speed.png', dpi=300)

# plt.savefig('temp/mnist_speed.png', dpi=300)
# plt.show()
