DEHB versions:

- **Version 1.0**
    - Each DEHB iteration begins with a random population sampled for 
    the population size determined by Successive Halving (SH).
    - DE is run for _5 generations_ for each SH iteration inside each
    such DEHB iteration. 

![v1](utils/misc/flow_v1.png)

- **Version 1.1**
    - In addition to Version 1.0, the population from the highest budget 
    is collected in a global population whose size is determined by the the 
    pop size determined for full budget, summed over the number of SH brackets.
    - After the global pop size has been populated, individuals for mutation are 
    sampled from the global population.
    - When the full budget population is evolved, that population replaces the 
    weakest individuals from the global population.
    
    
- **Version 2**
    - Only the first DEHB iteration's first SH iteration is initialized
     with a random population.
    - In each of the SH iterations, the top _n%_ (as selected by SH 
    budget spacing) is passed on to the next budget.
    - DE is performed on this population for _1 generation_ and the
    single large population is updated with these evolved individuals.
    - The next SH iteration then selects the top _n%_ from the evolved
    individuals from the previous SH steps and so on.
    
![v2](utils/misc/flow_v2.png)


- **Version 3.0**
    - The total number of budget/configuration spacing from SH is
    determined at the beginning and that many sets of population are
    maintained, i.e., each population set consists of individuals 
    evaluated on the same budget.
    - These population sets are created during the first DEHB iteration.
    - For any j-th SH iteration (j>1) in any i-th DEHB iteration (i>1),
    the top _n%_ individuals are forwarded to the j+1-th SH iteration.
    - These individuals are evaluated for the budget for j+1-th SH step
    and then put into a pool along with the current population from 
    (i-1)-th DEHB iteration for the corresponding budget, and ranked 
    where the top half is retained.
    - These individuals undergo DE for _1 generation_ and the top _n %_
    is forwarded to the next step and so on.  

![v3](utils/misc/flow_v3.png)


- **Version 3.1**
    - Proceeds as Version 3.0, however maintains a global population created
    from the populations evaluated on full budget, as candidates parents for 
    mutation, just as Version 1.1.
    
- **Version 3.2**
    - Exactly the same as Version 3.0, but without the ranked selection (point 4)
    procedure, where two sets of populations are jointly ranked for selection.
    - The top _n%_ individuals that are forwarded to the j+1-th SH 
    iteration, is yet evaluated on the higher budget. Instead they are kept as 
    candidate population for mutation in the j+1-th iteration.
    - The current population for the higher budget serves as the trial population, 
    while the population forwarded from the previous budget is the sample space for
    mutation.    
    
- **Version 4.0**
    - Exactly like Version 3.0 with one change of number of generations not being 1. 
    Number of generations to evolve at each budget is not a hyperparameter to DEHB but 
    is an internal parameter, also determined by Successive Halving.
    - Along with the original SH allocations for budgets and pop sizes, another tweaked
    SH is used to yield a factor _eta_ less for the number of samples. This is used as 
    generations to evolve for each budget.
    - In effect, the pop sizes and number of generations are correlated, pop size at 
    each SH iteration is _eta_ times more than the number of generations.
    - Rest of the details of S
    
- **Version 4.1**
    - Version 3.1 with the number of generations from SH change of Version 4.0
    
- **Version 4.2**
    - Version 3.2 with the number of generations from SH change of Version 4.0
