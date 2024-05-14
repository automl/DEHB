### Basic single worker setup
A basic setup for optimizing can be done as follows. Please note, that this is example should solely show a simple setup of `dehb`. More in-depth examples can be found in the [examples folder](https://github.com/automl/DEHB/tree/master/examples). First we need to setup a `ConfigurationSpace`, from which Configurations will be sampled:

```python exec="true" source="material-block" result="python" title="Configuration Space" session="someid"
from ConfigSpace import ConfigurationSpace, Configuration

cs = ConfigurationSpace({"x0": (3.0, 10.0), "x1": ["red", "green"]})
print(cs)
```

Next, we need an `object_function`, which we are aiming to optimize:
```python exec="true" source="material-block" result="python" title="Configuration Space" session="someid"
import numpy as np

def objective_function(x: Configuration, fidelity: float, **kwargs):
    # Replace this with your actual objective value (y) and cost.
    cost = (10 if x["x1"] == "red" else 100) + fidelity
    y = x["x0"] + np.random.uniform()
    return {"fitness": y, "cost": x["x0"]}

sample_config = cs.sample_configuration()
print(sample_config)

result = objective_function(sample_config, fidelity=10)
print(result)
```

Finally, we can setup our optimizer and run DEHB:

```python exec="true" source="material-block" result="python" title="Configuration Space" session="someid"
from dehb import DEHB

dim = len(cs.get_hyperparameters())
optimizer = DEHB(
    f=objective_function,
    cs=cs,
    dimensions=dim,
    min_fidelity=3,
    max_fidelity=27,
    eta=3,
    n_workers=1,
    output_path="./logs",
)

# Run optimization for 1 bracket. Output files will be saved to ./logs
traj, runtime, history = optimizer.run(brackets=1, verbose=True)
config_id, config, fitness, runtime, fidelity, _ = history[0]
print("config", config)
print("fitness", fitness)
print("runtime", runtime)
print("fidelity", fidelity)
```