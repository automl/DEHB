## Running DEHB using Ask & Tell or built-in run function
### Introduction
DEHB allows users to either utilize the Ask & Tell interface for manual task distribution or leverage the built-in functionality (`run`) to set up a Dask cluster autonomously. DEHB aims to minimize the objective function (`f=`) specified by the user, thus this function play a central role in the optimization. In the following we aim to give an overview about the arguments the objective function must have and how the structure of the results should look like.

### The Objective Function
The objective function needs to have the parameters `config` and `fidelity` and evaluate the given configuration on the given fidelity. In a neural network optimization context, the fidelity could e.g. be the number of epochs to run the hyperparameter configuration for.

Let us now have a look at what the objective function should return. DEHB expects to receive a results `dict` from the objective function. has to contain the keys `fitness` and `cost`. `fitness` resembles the objective you are trying to optimize, e.g. validation loss. `cost` resembles the computational cost for computing the result, e.g. the wallclock time for training and validating a neural network to achieve the validation loss specified in `fitness`. It is also possible to add the field `info` to the `result` in order to store additional, user-specific information.

!!! note "User-specific information `info`"

Please note, that we only support types, that are serializable by `pandas`. If
non-serializable types are used, DEHB will not be able to save the history.
If you want to be on the safe side, please use built-in python types.

Now that we have cleared up what the inputs and outputs of the objective function should be, we will also provide you with a small example of what the objective function could look like. For a complete example, please have a look at one of our [examples](../examples/01.1_Optimizing_RandomForest_using_DEHB.ipynb).

```python
def your_objective_function(config, fidelity):
    val_loss, val_accuracy, time_taken = train_config_for_epochs(config, fidelity)
    
    # Note, that we use the validation loss as the feedback signal for DEHB, since we aim to minimize it
    return {
        "fitness": val_loss,    # mandatory
        "cost": time_taken,     # mandatory
        "info": {               # optional
            "validation_accuracy": val_acc
        }
    }
```

### Run Function
To utilize the `run` function, simply setup DEHB as you prefer and then call `dehb.run` with your specified compute budget:

```python
optimizer = DEHB(
    f=your_objective_function,
    cs=config_space, 
    dimensions=dimensions, 
    min_fidelity=min_fidelity, 
    max_fidelity=max_fidelity)

optimizer.run(fevals=20) # Run for 20 function evaluations
```

### Ask & Tell
The Ask & Tell functionality can be utilized as follows:

```python
optimizer = DEHB(
    f=your_objective_function, # Here we do not need to necessarily specify the objective function, but it can still be useful to call 'run' later.
    cs=config_space, 
    dimensions=dimensions, 
    min_fidelity=min_fidelity, 
    max_fidelity=max_fidelity)

# Ask for next configuration to run
job_info = optimizer.ask()

# Run the configuration for the given fidelity. Here you can freely distribute the computation to any worker you'd like.
result = your_objective_function(config=job_info["config"], fidelity=job_info["fidelity"])

# When you received the result, feed them back to the optimizer
optimizer.tell(job_info, result)
```
