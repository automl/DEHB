### Using the Ask & Tell interface
DEHB allows users to either utilize the Ask & Tell interface for manual task distribution or leverage the built-in functionality (`run`) to set up a Dask cluster autonomously.
The Ask & Tell functionality can be utilized as follows:
```python
optimizer = DEHB(
    f=your_target_function, # Here we do not need to necessarily specify the target function, but it can still be useful to call 'run' later.
    cs=config_space, 
    dimensions=dimensions, 
    min_fidelity=min_fidelity, 
    max_fidelity=max_fidelity)

# Ask for next configuration to run
job_info = optimizer.ask()

# Run the configuration for the given fidelity. Here you can freely distribute the computation to any worker you'd like.
result = your_target_function(config=job_info["config"], fidelity=job_info["fidelity"])

# When you received the result, feed them back to the optimizer
optimizer.tell(job_info, result)
```