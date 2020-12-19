'''Runs DEHB on NAS-Bench-101 and NAS-HPO-Bench
'''

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks/'))
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks-development/'))

sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks_development/'))
sys.path.append('../NAS101/nasbench')

import json
import time
import argparse
import numpy as np
import pandas as pd

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C

import autogluon as ag

from dehb import DEHB


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


parser = argparse.ArgumentParser()
choices = ["protein_structure", "slice_localization", "naval_propulsion",
           "parkinsons_telemonitoring", "nas_cifar10a", "nas_cifar10b", "nas_cifar10c"]
parser.add_argument('--benchmark', default="protein_structure", type=str,
                    help="specify the benchmark to run on from among {}".format(choices))
parser.add_argument('--data_dir', default="../nas_benchmarks_development/"
                                          "tabular_benchmarks/fcnet_tabular_benchmarks/",
                    type=str, nargs='?', help='specifies the path to the tabular data')
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
    folder = "abohb"
else:
    folder = args.folder

output_path = os.path.join(args.output_path, folder)
os.makedirs(output_path, exist_ok=True)

bench_type = None

if args.benchmark == "nas_cifar10a":  # NAS-Bench-101
    min_budget = 4
    max_budget = 108
    b = NASCifar10A(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "nas_cifar10b":  # NAS-Bench-101
    min_budget = 4
    max_budget = 108
    b = NASCifar10B(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "nas_cifar10c":  # NAS-Bench-101
    min_budget = 4
    max_budget = 108
    b = NASCifar10C(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "protein_structure":  # NAS-HPO-Bench
    bench_type = 'nas-hpo'
    min_budget = 3
    max_budget = 100
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "slice_localization":  # NAS-HPO-Bench
    bench_type = 'nas-hpo'
    min_budget = 3
    max_budget = 100
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "naval_propulsion":  # NAS-HPO-Bench
    bench_type = 'nas-hpo'
    min_budget = 3
    max_budget = 100
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "parkinsons_telemonitoring":  # NAS-HPO-Bench
    bench_type = False
    min_budget = 'nas-hpo'
    min_budget = 3
    max_budget = 100
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

cs = b.get_configuration_space()
dehb = DEHB(min_budget=min_budget, max_budget=max_budget, eta=3)
budgets = dehb.budgets
brackets = 1


if 'cifar10a' in args.benchmark:
    @ag.args(
        edge_0=ag.space.Categorical('0', '1'),
        edge_1=ag.space.Categorical('0', '1'),
        edge_10=ag.space.Categorical('0', '1'),
        edge_11=ag.space.Categorical('0', '1'),
        edge_12=ag.space.Categorical('0', '1'),
        edge_13=ag.space.Categorical('0', '1'),
        edge_14=ag.space.Categorical('0', '1'),
        edge_15=ag.space.Categorical('0', '1'),
        edge_16=ag.space.Categorical('0', '1'),
        edge_17=ag.space.Categorical('0', '1'),
        edge_18=ag.space.Categorical('0', '1'),
        edge_19=ag.space.Categorical('0', '1'),
        edge_2=ag.space.Categorical('0', '1'),
        edge_20=ag.space.Categorical('0', '1'),
        edge_3=ag.space.Categorical('0', '1'),
        edge_4=ag.space.Categorical('0', '1'),
        edge_5=ag.space.Categorical('0', '1'),
        edge_6=ag.space.Categorical('0', '1'),
        edge_7=ag.space.Categorical('0', '1'),
        edge_8=ag.space.Categorical('0', '1'),
        edge_9=ag.space.Categorical('0', '1'),
        op_node_0=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        op_node_1=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        op_node_2=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        op_node_3=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        op_node_4=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        budget=max_budget,
        budget_list=budgets
    )
    # Common objective function for DE representing NAS-Bench-101 & NAS-HPO-Bench
    def f(args, reporter):
        config = cs.sample_configuration()
        for hyper in cs.get_hyperparameters():
            config[hyper.name] = args[hyper.name]
        for budget in args.budget_list:
            # budget = args.budget_list[i]
            if budget is not None:
                fitness, cost = b.objective_function(config, budget=int(budget))
            else:
                fitness, cost = b.objective_function(config)
            test_fitness, _ = b.objective_function_test(config)['function_value']

            reporter(budget=budget, score=-fitness, test_score=-test_fitness, cost=cost)
elif 'cifar10b' in args.benchmark:
    @ag.args(
        edge_0=ag.space.Categorical('0', '1'),
        edge_1=ag.space.Categorical('0', '1'),
        edge_2=ag.space.Categorical('0', '1'),
        edge_3=ag.space.Categorical('0', '1'),
        edge_4=ag.space.Categorical('0', '1'),
        edge_5=ag.space.Categorical('0', '1'),
        edge_6=ag.space.Categorical('0', '1'),
        edge_7=ag.space.Categorical('0', '1'),
        edge_8=ag.space.Categorical('0', '1'),
        op_node_0=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        op_node_1=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        op_node_2=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        op_node_3=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        op_node_4=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        budget=max_budget,
        budget_list=budgets
    )
    # Common objective function for DE representing NAS-Bench-101 & NAS-HPO-Bench
    def f(args, reporter):
        config = cs.sample_configuration()
        for hyper in cs.get_hyperparameters():
            config[hyper.name] = args[hyper.name]
        for budget in args.budget_list:
            # budget = args.budget_list[i]
            if budget is not None:
                fitness, cost = b.objective_function(config, budget=int(budget))
            else:
                fitness, cost = b.objective_function(config)
            test_fitness, _ = b.objective_function_test(config)['function_value']

            reporter(budget=budget, score=-fitness, test_score=-test_fitness, cost=cost)
elif 'cifar10c' in args.benchmark:
    @ag.args(
        edge_0=ag.space.Real(0, 1, default=0.5),
        edge_1=ag.space.Real(0, 1, default=0.5),
        edge_10=ag.space.Real(0, 1, default=0.5),
        edge_11=ag.space.Real(0, 1, default=0.5),
        edge_12=ag.space.Real(0, 1, default=0.5),
        edge_13=ag.space.Real(0, 1, default=0.5),
        edge_14=ag.space.Real(0, 1, default=0.5),
        edge_15=ag.space.Real(0, 1, default=0.5),
        edge_16=ag.space.Real(0, 1, default=0.5),
        edge_17=ag.space.Real(0, 1, default=0.5),
        edge_18=ag.space.Real(0, 1, default=0.5),
        edge_19=ag.space.Real(0, 1, default=0.5),
        edge_2=ag.space.Real(0, 1, default=0.5),
        edge_20=ag.space.Real(0, 1, default=0.5),
        edge_3=ag.space.Real(0, 1, default=0.5),
        edge_4=ag.space.Real(0, 1, default=0.5),
        edge_5=ag.space.Real(0, 1, default=0.5),
        edge_6=ag.space.Real(0, 1, default=0.5),
        edge_7=ag.space.Real(0, 1, default=0.5),
        edge_8=ag.space.Real(0, 1, default=0.5),
        edge_9=ag.space.Real(0, 1, default=0.5),
        num_edges=ag.space.Int(0, 9, default=4),
        op_node_0=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        op_node_1=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        op_node_2=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        op_node_3=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        op_node_4=ag.space.Categorical('conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3'),
        budget=max_budget,
        budget_list=budgets
    )
    # Common objective function for DE representing NAS-Bench-101 & NAS-HPO-Bench
    def f(args, reporter):
        config = cs.sample_configuration()
        for hyper in cs.get_hyperparameters():
            config[hyper.name] = args[hyper.name]
        for budget in args.budget_list:
            # budget = args.budget_list[i]
            if budget is not None:
                fitness, cost = b.objective_function(config, budget=int(budget))
            else:
                fitness, cost = b.objective_function(config)
            test_fitness, _ = b.objective_function_test(config)['function_value']

            reporter(budget=budget, score=-fitness, test_score=-test_fitness, cost=cost)
else:  # bench_type == 'nas-hpo'
    @ag.args(
        activation_fn_1=ag.space.Categorical('tanh', 'relu'),
        activation_fn_2=ag.space.Categorical('tanh', 'relu'),
        batch_size=ag.space.Categorical('8', '16', '32', '64'),
        dropout_1=ag.space.Categorical('0.0', '0.3', '0.6'),
        dropout_2=ag.space.Categorical('0.0', '0.3', '0.6'),
        init_lr=ag.space.Categorical('0.0005', '0.001', '0.005', '0.01', '0.1'),
        lr_schedule=ag.space.Categorical('cosine', 'const'),
        n_units_1=ag.space.Categorical('16', '32', '64', '128', '256', '512'),
        n_units_2=ag.space.Categorical('16', '32', '64', '128', '256', '512'),
        budget=max_budget,
        budget_list=budgets
    )
    # Common objective function for DE representing NAS-Bench-101 & NAS-HPO-Bench
    def f(args, reporter):
        config = cs.sample_configuration()
        for hyper in cs.get_hyperparameters():
            config[hyper.name] = args[hyper.name]
        for budget in args.budget_list:
            # budget = args.budget_list[i]
            if budget is not None:
                fitness, cost = b.objective_function(config, budget=int(budget))
            else:
                fitness, cost = b.objective_function(config)
            test_fitness, _ = b.objective_function_test(config)['function_value']

            reporter(budget=budget, score=-fitness, test_score=-test_fitness, cost=cost)


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
