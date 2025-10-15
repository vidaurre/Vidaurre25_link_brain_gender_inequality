# Vidaurre25_link_brain_gender_inequality
This repository contains the Python code for the paper "The link between gender inequality and the distribution of brain regions' relative sizes across the lifespan and the world"

The pipeline can be run as:

```
sh run_all.sh
```

This runs all the analyses to create the results reported in the paper.  
We refer to the script for reference to the various steps of the analysis, each of which is performed by a separate Python script. 
The purpose of each script is documented in the header of the script.

The first parameter of each Python script with run_all.sh (excepting run_ttests.py, where it is the second) refers to whether we run the script on real data. 
The second parameter (excepting run_ttests, where it is the third) refers to whether we run a quick demo run as a sanity check (for example, with only a few permutations in the permutation testing). 

Unfortunately, due to legal requirements, the brain data cannot be shared, although it can be requested formally to the data owners. 
The socioeconomic data is however public and it is included in the repository. 
For this reason, and just for illustration purposes, we provide the option of running the pipeline on synthetic data.
This is generated using generate_random_data.py, but the results cannot be reproduced as these data have no structure. 

For running the pipeline in quick demo on synthetic data:

```
sh run_fast.sh
```

## Dependencies 

The required dependencies to use the code are:

- scikit-learn
- scipy
- matplotlib
- pandas
- country_converter
- shap

Two ad-hoc libraries are included within the code: subset_optimisation.py (for sampling) and func_testing.py (for testing). 

The code was run on the 3.12 version of Python

## Reference 

The paper can be cited as 

Vidaurre, D., Morell-Ortega, S., Oyarzo, P., Butler, C.R., Charquero-Ballester, M., Gadea, M., Morandat, F., Mansencal, B., Coupe, P. and Manjon, J.V., 2025. The link between gender inequality and the distribution of brain regions' relative sizes across the lifespan and the world. bioRxiv, pp.2025-09.


