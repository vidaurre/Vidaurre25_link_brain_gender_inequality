# Vidaurre25_link_brain_gender_inequality
This repository contains the code for the paper "The link between gender inequality and the distribution of brain regions' relative sizes across the lifespan and the world""

The pipeline can be run as:

```
sh run_all.sh
```

This runs all the scripts to create the results reported in the paper.  
The first parameter of each script (excepting run_ttests, where it is the second) refers to whether we run the script on real data. 
The second parameter (excepting run_ttests, where it is the third) refers to whether we run a quick demo run as a sanity check (for example, with only a few permutations in the permutation testing). 

Unfortunately, due to legal requirements, the brain data cannot be shared, although it can be requested formally to the data owners. 
The socioeconomic data is however public and it is included in the repository. 
For this reason, and just for illustration purposes, we provide the option of running the pipeline on synthetic data.
This is generated using generate_random_data.py, but the results cannot be reproduced as these data have no structure. 

For running the pipeline in quick demo on synthetic data:

```
sh run_fast.sh
```

The paper can be cited as 

Vidaurre, D., Morell-Ortega, S., Oyarzo, P., Butler, C.R., Charquero-Ballester, M., Gadea, M., Morandat, F., Mansencal, B., Coupe, P. and Manjon, J.V., 2025. The link between gender inequality and the distribution of brain regions' relative sizes across the lifespan and the world. bioRxiv, pp.2025-09.


