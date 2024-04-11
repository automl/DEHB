### DEHB Hyperparameters

*We recommend the default settings*.
The default settings were chosen based on ablation studies over a collection of diverse problems 
and were found to be *generally* useful across all cases tested. 
However, the parameters are still available for tuning to a specific problem.

The Hyperband components:

- *min\_fidelity*: Needs to be specified for every DEHB instantiation and is used in determining 
the fidelity spacing for the problem at hand.
- *max\_fidelity*: Needs to be specified for every DEHB instantiation. Represents the full-fidelity 
evaluation or the actual black-box setting.
- *eta*: (default=3) Sets the aggressiveness of Hyperband's aggressive early stopping by retaining
1/eta configurations every round
  
The DE components:

- *strategy*: (default=`rand1_bin`) Chooses the mutation and crossover strategies for DE. `rand1` 
represents the *mutation* strategy while `bin` represents the *binomial crossover* strategy. \
  Other mutation strategies include: {`rand2`, `rand2dir`, `best`, `best2`, `currenttobest1`, `randtobest1`}\
  Other crossover strategies include: {`exp`}\
  Mutation and crossover strategies can be combined with a `_` separator, for e.g.: `rand2dir_exp`.
- *mutation_factor*: (default=0.5) A fraction within [0, 1] weighing the difference operation in DE
- *crossover_prob*: (default=0.5) A probability within [0, 1] weighing the traits from a parent or the mutant