#!/bin/sh

python generate_data_spin_model.py --seed -1 --size 10 --num_beta 10 --rep 100000 --BASE_DIR ~/test
python calculate_Fig1a.py          --size 10 --num_beta 10 --rep 100000 --BASE_DIR ~/test

