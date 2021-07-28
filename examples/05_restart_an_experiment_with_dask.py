"""
Before calling this event start the scheduler with

`dask-scheduler --scheduler-file <result_path>/scheduler_file.txt`

and then the worker with

`dask-worker --scheduler-file <result_path>/scheduler_file.txt` --name 1 --resources "limit_proc=1" --no-nanny

Note that we give the worker a resource. By doing so, only a single task can be executed per worker.

"""

import time
import logging
logging.basicConfig(level=logging.INFO)

from distributed import Client
from pathlib import Path
from dehb.optimizers.dehb_with_warmstart import DEHB

from hpobench.container.benchmarks.surrogates.paramnet_benchmark import ParamNetPokerOnTimeBenchmark


def objective_function(config, budget, **kwargs):

    start = time.time()
    socket_id = kwargs.get('socket_id')
    benchmark = ParamNetPokerOnTimeBenchmark(socket_id=socket_id)
    result_dict = benchmark.objective_function(configuration=config, fidelity={'budget': int(budget)})
    finish = time.time()
    return {'fitness': result_dict['function_value'], 'cost': result_dict['cost'],
            'info': {'res_info': result_dict['info'], 'time': float(finish - start)}}


def main(result_path: str, seed=0):

    result_path = Path(result_path)
    result_path.mkdir(exist_ok=True, parents=True)

    checkpoint_file = result_path / 'checkpoint.pkl'
    scheduler_file = result_path / 'scheduler_file.txt'

    benchmark = ParamNetPokerOnTimeBenchmark()

    client = Client(scheduler_file=scheduler_file)

    dehb = DEHB(f=objective_function,
                cs=benchmark.get_configuration_space(seed=seed),
                dimensions=len(benchmark.get_configuration_space().get_hyperparameters()),
                min_budget=81,  # Those are the budgets used by the benchmark.
                max_budget=2187,
                eta=3,
                output_path=result_path / 'dehb_logs',
                client=client,
                # Limit the tasks per worker by starting the worker with the same resource! See above.
                client_resources={'limit_proc': 1},
                )

    try:
        traj, runtime, history = dehb.run(total_cost=20,  # Let the procedure run for 20 seconds.
                                          verbose=True,
                                          save_intermediate=True,
                                          # arguments below are part of **kwargs shared across workers
                                          eta=3,
                                          result_path=result_path,
                                          socket_id=benchmark.socket_id)
    except Exception:
        # One could this a try-except to save the intermediate results in case of an error.
        dehb.save_checkpoint(checkpoint_file)

    # Call this function to save the checkpoint to disk.
    dehb.save_checkpoint(checkpoint_file)

    dehb_2 = DEHB(f=objective_function,
                  cs=benchmark.get_configuration_space(seed=seed),
                  dimensions=len(benchmark.get_configuration_space().get_hyperparameters()),
                  min_budget=81,
                  max_budget=2187,
                  eta=3,
                  output_path=result_path / 'dehb_logs',
                  client=client,
                  client_resources={'limit_proc': 1},  # Also the new object needs this limit.
                  checkpoint_file=result_path / 'checkpoint.pkl',
                  restore_checkpoint=True,
                  )

    traj2, runtime2, history2 = dehb_2.run(total_cost=20 + 20,
                                           verbose=True,
                                           save_intermediate=True,
                                           # arguments below are part of **kwargs shared across workers
                                           socket_id=benchmark.socket_id)


    from matplotlib import pyplot as plt
    import numpy as np

    f = plt.figure()
    plt.plot(np.arange(len(traj2)), traj2, label='restarted')
    plt.plot(np.arange(len(traj)), traj, label='first run')
    plt.yscale('log')
    plt.legend()
    plt.savefig(result_path / 'traj.png')
    plt.close()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str)
    args = parser.parse_args()

    main(result_path=args.result_path, seed=0)