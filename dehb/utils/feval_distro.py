import numpy as np
from matplotlib import pyplot as plt

title = ""

def compute_stats(traj, runtime, history):
    counts  = {}
    for i in range(len(history)):
        _, _, budget = history[i]
        if str(budget) in counts.keys():
            counts[str(budget)] += 1
        else:
            counts[str(budget)] = 1
    counts['runtime'] = np.log10(np.cumsum(runtime)[-1])
    return counts


counts = compute_stats(traj, runtime, history)
x = list(counts.keys())
height = [counts[key] for key in counts]

ax = plt.gca()
ax.bar(x, height)
ax.set_xticklabels(counts.keys())
for key in counts.keys():
    ax.text(key, counts[key] + 20, str(counts[key]), ha="center", va="center")
    val = np.log10(float(key) * counts[key])
    ax.text(key, counts[key] - 20, "{:<0.3f}".format(val), ha="center", va="center")
ax.text(list(counts.keys())[-2], max([counts[key] for key in counts.keys()]) - 500,
        "Incumbent: {:<0.5f}\nTotal evals: {}\nCum. runtime: {:<0.2f}".format(traj[-1], len(traj), np.log10(np.cumsum(runtime)[-1])), ha="center", va="center")
plt.title(title)
plt.show()