import time
import logging
logging.basicConfig(level=logging.INFO)

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

    benchmark = ParamNetPokerOnTimeBenchmark()

    dehb = DEHB(f=objective_function,
                cs=benchmark.get_configuration_space(seed=seed),
                dimensions=len(benchmark.get_configuration_space().get_hyperparameters()),
                min_budget=81,  # Those are the budgets used by the benchmark.
                max_budget=2187,
                eta=3,
                output_path=result_path / 'dehb_logs',
                n_workers=1)

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

    # To restart now the optimization procedure, create a new object.
    # Set the parameters `checkpoint_file` and `restore_checkpooint`. Then, the checkpoint will be automatically loaded.
    dehb_2 = DEHB(f=objective_function,
                  cs=benchmark.get_configuration_space(seed=seed),
                  dimensions=len(benchmark.get_configuration_space().get_hyperparameters()),
                  min_budget=81,
                  max_budget=2187,
                  eta=3,
                  output_path=result_path / 'dehb_logs',
                  n_workers=1,
                  checkpoint_file=result_path / 'checkpoint.pkl',
                  restore_checkpoint=True)

    # NOTE: Make sure to increase the time limit!!
    traj2, runtime2, history2 = dehb_2.run(total_cost=20 + 20,
                                           verbose=True,
                                           save_intermediate=True,
                                           # This parameter is needed for the HPOBench-Benchmark Object.
                                           socket_id=benchmark.socket_id)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str)
    args = parser.parse_args()

    main(result_path=args.result_path, seed=0)