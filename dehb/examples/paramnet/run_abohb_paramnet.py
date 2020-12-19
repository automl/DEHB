import os
import json
import time
import argparse
import numpy as np
import pandas as pd

import autogluon as ag

from hpolib.benchmarks.surrogates.paramnet import SurrogateReducedParamNetTime

from dehb import DEHB


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', help="name of the dataset used",
                    choices=['adult', 'higgs', 'letter', 'mnist', 'optdigits', 'poker'])
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--workers', default=1, type=int, nargs='?', help='number of workers')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--iter', default=20, type=int, nargs='?',
                    help='number of DEHB iterations')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--folder', default=None, type=str, nargs='?',
                    help='name of folder where files will be dumped')

args = parser.parse_args()

if args.folder is None:
    folder = "{}/abohb".format(args.dataset)
else:
    folder = "{}/{}".format(args.dataset, args.folder)

output_path = os.path.join(args.output_path, folder)
os.makedirs(output_path, exist_ok=True)

dataset = args.dataset
b = SurrogateReducedParamNetTime(dataset=dataset)

# Parameter space to be used by DE
cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

budgets = {  # (min, max)-budget (seconds) for the different data sets
    'adult': (9, 243),
    'higgs': (9, 243),
    'letter': (3, 81),
    'mnist': (9, 243),
    'optdigits': (1, 27),
    'poker': (81, 2187),
}
min_budget, max_budget = budgets[dataset]
dehb = DEHB(min_budget=min_budget, max_budget=max_budget, eta=3)
budgets = dehb.budgets
brackets = 1

# all datasets share same param space
@ag.args(
    x0=ag.space.Real(-6, -2, default=-4),
    x1=ag.space.Real(3, 8, default=5.5),
    x2=ag.space.Real(4, 8, default=6),
    x3=ag.space.Real(-4, 0, default=-2),
    x4=ag.space.Real(1, 5, default=3),
    x5=ag.space.Real(0, 0.5, default=0.25),
    budget=max_budget,
    budget_list=budgets
)
def f(args, reporter):
    global b, cs
    config = cs.sample_configuration()
    for hyper in cs.get_hyperparameters():
        config[hyper.name] = args[hyper.name]
    for budget in args.budget_list:
        # budget = args.budget_list[i]
        if budget is not None:
            fitness = b.objective_function(config, budget=budget)
        else:
            fitness = b.objective_function(config)
            budget = max_budget
        fitness = fitness['function_value']
        cost = budget
        test_fitness = b.objective_function_test(config)['function_value']

        reporter(budget=budget, score=-fitness, test_score=-test_fitness, cost=cost)


def save_to_json(training_history, output_path, run_id):
    task_dfs = []
    for task_id in training_history:
        task_df = pd.DataFrame(training_history[task_id])
        task_df = task_df.assign(
            task_id=task_id,
            target_epoch=task_df["budget"].iloc[-1],
            y=-task_df['score'],
            y_test=-task_df['test_score']
        )
        task_dfs.append(task_df)

    result = pd.concat(task_dfs, axis="index", ignore_index=True, sort=True)

    # re-order by runtime
    result = result.sort_values(by="time_since_start")
    result = result.assign(x=np.cumsum(result['cost']))

    # calculate incumbent best -- the cumulative minimum of the error.
    result = result.assign(best=result["y"].cummin())
    test_inc_scores = []
    test_inc = result['y_test'][0]
    for i in range(result.shape[0]):
        if result['y'][i] == result['best'][i]:  # incumbent is updated, so new test score
            test_inc = result['y_test'][i]
        test_inc_scores.append(test_inc)
    result = result.assign(best_test=test_inc_scores)

    res = dict()
    res['validation_score'] = result["best"].tolist()
    res['test_score'] = result["best_test"].tolist()
    res['runtime'] = result["x"].tolist()
    fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
    json.dump(res, fh)
    fh.close()


scheduler = ag.scheduler.HyperbandScheduler(
    f,  # train_fn,
    searcher='bayesopt',
    resource={'num_cpus': args.workers, 'num_gpus': 0},
    num_trials=args.iter,
    reward_attr='score',  # metric maximized
    time_attr='budget',  # specifying fidelity/budget
    grace_period=min_budget,  # minimum budget/resource
    max_t=max_budget,  # maximum budget/resource
    brackets=brackets,
    reduction_factor=3,  # eta
    # keep_size_ratios=True,
    # maxt_pending=True
)

start = time.time()
scheduler.run()
print("Time taken: ", time.time() - start)

save_to_json(scheduler.training_history, output_path, args.run_id)
