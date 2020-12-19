import os
import time
import json
import pickle
import argparse
import numpy as np
import pandas as pd

import autogluon as ag

from hpolib.benchmarks.synthetic_functions.counting_ones import CountingOnes

from dehb import DEHB


def calc_regrets(history):
    global de, b, cs
    d = len(cs.get_hyperparameters())
    valid_scores = []
    test_scores = []
    test_regret = 1
    valid_regret = 1
    inc = np.inf
    for i in range(len(history)):
        valid_regret = (history[i][1] + d) / d
        if valid_regret < inc:
            inc = valid_regret
            config = de.vector_to_configspace(history[i][0])
            res = b.objective_function_test(config)
            test_regret = (res['function_value'] + d) / d
        test_scores.append(test_regret)
        valid_scores.append(inc)
    return valid_scores, test_scores


def save_json(valid, test, runtime, output_path, run_id):
    res = {}
    res['regret_validation'] = valid
    res['regret_test'] = test
    res['runtime'] = np.cumsum(runtime).tolist()
    fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
    json.dump(res, fh)
    fh.close()


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


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
    result['y'] = (result['y'] + dimensions) / dimensions
    result['y_test'] = (result['y_test'] + dimensions) / dimensions

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


parser = argparse.ArgumentParser()
parser.add_argument('--n_cont', default=4, type=int, nargs='?',
                    help='number of continuous variables')
parser.add_argument('--n_cat', default=4, type=int, nargs='?',
                    help='number of categorical variables')
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

dim_folder = "{}+{}".format(args.n_cont, args.n_cat)

if args.folder is None:
    folder = "{}/abohb".format(dim_folder)
else:
    folder = "{}/{}".format(dim_folder, args.folder)

output_path = os.path.join(args.output_path, folder)
os.makedirs(output_path, exist_ok=True)

# Loading benchmark
b = CountingOnes()

# Parameter space to be used by DE
cs = b.get_configuration_space(n_continuous=args.n_cont, n_categorical=args.n_cat)
dimensions = len(cs.get_hyperparameters())

min_budget = 576 // dimensions
max_budget = 93312 // dimensions

y_star_test = -dimensions  # incorporated in regret_calc as normalized regret: (f(x) + d) / d

dehb = DEHB(min_budget=min_budget, max_budget=max_budget, eta=3)
budgets = np.array(dehb.budgets, dtype=int).tolist()
brackets = 1
print(min_budget, max_budget, budgets)


@ag.args(
    cat_0=ag.space.Categorical(0, 1),
    cat_1=ag.space.Categorical(0, 1),
    cat_2=ag.space.Categorical(0, 1),
    cat_3=ag.space.Categorical(0, 1),
    float_0=ag.space.Real(0.0, 1.0, default=0.5),
    float_1=ag.space.Real(0.0, 1.0, default=0.5),
    float_2=ag.space.Real(0.0, 1.0, default=0.5),
    float_3=ag.space.Real(0.0, 1.0, default=0.5),
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


scheduler = ag.scheduler.HyperbandScheduler(
    f,  # train_fn,
    searcher='bayesopt',
    resource={'num_cpus': args.workers, 'num_gpus': 0},
    num_trials=args.iter,
    reward_attr='score',  # metric maximized
    time_attr='budget',  # specifying fidelity/budget
    grace_period=budgets[0],  # minimum budget/resource
    max_t=budgets[-1],  # maximum budget/resource
    brackets=brackets,
    reduction_factor=3,  # eta
    # keep_size_ratios=True,
    # maxt_pending=True
)

start = time.time()
scheduler.run()
print("Time taken: ", time.time() - start)

save_to_json(scheduler.training_history, output_path, args.run_id)
