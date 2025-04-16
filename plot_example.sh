#!/bin/sh

python generate_data_spin_model.py --size 10 --num_beta 10
python calculate_Fig1a.py --num_beta 10 --size 10

