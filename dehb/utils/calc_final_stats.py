'''Script to obtain mean and standard deviation of the final test
    errors across runs for all algorithms'''

import os
import json
import sys
import pickle
import argparse
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fill_trajectory(performance_list, time_list, replace_nan=np.NaN):
    frame_dict = collections.OrderedDict()
    counter = np.arange(0, len(performance_list))
    for p, t, c in zip(performance_list, time_list, counter):
        if len(p) != len(t):
            raise ValueError("(%d) Array length mismatch: %d != %d" %
                             (c, len(p), len(t)))
        frame_dict[str(c)] = pd.Series(data=p, index=t)

    # creates a dataframe where the rows are indexed based on time
    # fills with NA for missing values for the respective timesteps
    merged = pd.DataFrame(frame_dict)
    # ffill() acts like a fillna() wherein a forward fill happens
    # only remaining NAs for in the beginning until a value is recorded
    merged = merged.ffill()

    performance = merged.values  # converts to a 2D numpy array
    time_ = merged.index.values        # retrieves the timestamps

    performance[np.isnan(performance)] = replace_nan

    if not np.isfinite(performance).all():
        raise ValueError("\nCould not merge lists, because \n"
                         "\t(a) one list is empty?\n"
                         "\t(b) the lists do not start with the same times and"
                         " replace_nan is not set?\n"
                         "\t(c) any other reason.")

    return performance, time_


parser = argparse.ArgumentParser()

parser.add_argument('--bench', default='101', type=str, nargs='?',
                    choices=['101', '1shot1', '201'], help='select benchmark')
parser.add_argument('--ssp', default=None, type=str, nargs='?')
parser.add_argument('--path', default='./', type=str, nargs='?',
                    help='path to encodings or jsons for each algorithm')
parser.add_argument('--folder', default='./', type=str, nargs='?',
                    help='path to where runs reside')
parser.add_argument('--n_runs', default=500, type=int, nargs='?',
                    help='number of runs to plot data for')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the plot will be saved')
parser.add_argument('--limit', default=1e7, type=float, help='wallclock limit')
parser.add_argument('--regret', default='validation', type=str, choices=['validation', 'test'],
                    help='type of regret')

args = parser.parse_args()
path = args.path
n_runs = args.n_runs
regret_type = args.regret
benchmark = args.bench
ssp = args.ssp
output_path = args.output_path
folder = args.folder

if benchmark == '201' or benchmark == '1shot1':
    folder = ssp
if benchmark == '201' or benchmark == '101':
    path = os.path.join(path, folder)

if benchmark == '1shot1' and ssp is None:
    print("Specify \'--ssp\' from {1, 2, 3} for choosing the search space for NASBench-1shot1.")
    sys.exit()

if benchmark == '101':
    methods = [("random_search", "RS"),
               ("bohb", "BOHB"),
               ("hyperband", "HB"),
               ("tpe", "TPE"),
               ("regularized_evolution", "RE"),
               ("de_pop20", "DE")]
elif benchmark == '201':
    methods = [("random_search", "RS"),
               ("tpe", "TPE"),
               ("regularized_evolution", "RE"),
               ("de_pop20", "DE")]

else:
    methods = [("RS", "RS"),
               ("BOHB", "BOHB"),
               ("HB", "HB"),
               ("TPE", "TPE"),
               ("RE", "RE"),
               ("DE_pop20", "DE")]

if benchmark == '101':
    sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks/'))
    sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks-development/'))
    from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
        FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
    from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C
    data_dir = os.path.join(os.getcwd(), "../nas_benchmarks-development/"
                                         "tabular_benchmarks/fcnet_tabular_benchmarks/")
    if ssp == "nas_cifar10a":
        b = NASCifar10A(data_dir=data_dir, multi_fidelity=False)
        y_star_valid = b.y_star_valid
        y_star_test = b.y_star_test
        inc_config = None
    elif ssp == "nas_cifar10b":
        b = NASCifar10B(data_dir=data_dir, multi_fidelity=False)
        y_star_valid = b.y_star_valid
        y_star_test = b.y_star_test
        inc_config = None
    elif ssp == "nas_cifar10c":
        b = NASCifar10C(data_dir=data_dir, multi_fidelity=False)
        y_star_valid = b.y_star_valid
        y_star_test = b.y_star_test
        inc_config = None
    elif ssp == "protein_structure":
        b = FCNetProteinStructureBenchmark(data_dir=data_dir)
        inc_config, y_star_valid, y_star_test = b.get_best_configuration()
    elif ssp == "slice_localization":
        b = FCNetSliceLocalizationBenchmark(data_dir=data_dir)
        inc_config, y_star_valid, y_star_test = b.get_best_configuration()
    elif ssp == "naval_propulsion":
        b = FCNetNavalPropulsionBenchmark(data_dir=data_dir)
        inc_config, y_star_valid, y_star_test = b.get_best_configuration()
    elif ssp == "parkinsons_telemonitoring":
        b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=data_dir)
        inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif benchmark == '1shot1':    
    sys.path.append(os.path.join(os.getcwd(), '../nasbench/'))
    sys.path.append(os.path.join(os.getcwd(), '../nasbench-1shot1/'))
    from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
    from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
    from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
    search_space = eval('SearchSpace{}()'.format(ssp))
    y_star_valid, y_star_test, inc_config = (search_space.valid_min_error,
                                             search_space.test_min_error, None)

else:
    sys.path.append(os.path.join(os.getcwd(), '../nas201/'))
    sys.path.append(os.path.join(os.getcwd(), '../AutoDL-Projects/lib/'))
    from nas_201_api import NASBench201API as API
    from models import CellStructure, get_search_spaces
    data_dir = os.path.join(os.getcwd(), "../nas201/NAS-Bench-201-v1_0-e61699.pth")
    api = API(data_dir)
    def find_nas201_best(api, dataset):
        arch, y_star_test = api.find_best(dataset=dataset, metric_on_set='ori-test')
        _, y_star_valid = api.find_best(dataset=dataset, metric_on_set='x-valid')
        return 1 - (y_star_valid / 100), 1 - (y_star_test / 100)
    y_star_valid, y_star_test = find_nas201_best(api, ssp)


final_regrets = {}
# looping and calculating for all methods
for index, (m, label) in enumerate(methods):
    regret = []
    runtimes = []
    for k, i in enumerate(np.arange(n_runs)):
        try:
            if benchmark in ['101', '201']:
                res = json.load(open(os.path.join(path, m, "run_%d.json" % i)))
            else:
                res = json.load(open(os.path.join(path, m, str(ssp), "run_%d.json" % i)))
            no_runs_found = False
        except Exception as e:
            print(m, i, e)
            no_runs_found = True
            continue
        regret_key =  "regret_validation" if regret_type == 'validation' else "regret_test"
        runtime_key = "runtime"
        _, idx = np.unique(res[regret_key], return_index=True)
        idx.sort()
        regret.append(np.array(res[regret_key])[idx])
        runtimes.append(np.array(res[runtime_key])[idx])

    if not no_runs_found:
        # finds the latest time where the first measurement was made across runs
        t = np.max([runtimes[i][0] for i in range(len(runtimes))])
        te, time = fill_trajectory(regret, runtimes, replace_nan=1)

        idx = time.tolist().index(t)
        te = te[idx:, :]
        time = time[idx:]

        # Clips off all measurements after 10^7s
        idx = np.where(time < args.limit)[0]

        ## final regret
        te = te[idx][-1]
        y_star =  y_star_valid if regret_type == 'validation' else y_star_test
        te = te + y_star
        final_regrets[label] = (np.mean(te), np.std(te))
        print("{:<5}: {} ==> {} +- {}".format(label, regret_type, np.mean(te), np.std(te)))

os.makedirs(output_path, exist_ok=True)
with open(os.path.join(output_path, 'final_regrets_{}_{}.json'.format(benchmark, folder)), 'w') as f:
    json.dump(final_regrets, f)
print('File dumped for {}/{}'.format(benchmark, folder))
print('\n')
