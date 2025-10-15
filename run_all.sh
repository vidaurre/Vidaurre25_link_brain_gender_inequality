#!/bin/bash

python3.12 preprocess_socioeconomic_data.py
python3.12 preprocess_brain_data.py
python3.12 gather_demographics.py 1
python3.12 compute_between_sexes_distances.py 1
python3.12 predict_sex.py 1 0 
python3.12 compute_shap_values.py 1 0
python3.12 run_ftests.py 1 0
python3.12 run_ttests.py 0 1 0
python3.12 run_ttests.py 1 1 0
python3.12 run_ttests.py 2 1 0


