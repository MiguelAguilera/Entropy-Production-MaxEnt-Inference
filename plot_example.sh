#!/bin/sh
# 
ARGS="--patterns 10 --beta_min 1 --beta_max 4 --size 20 --num_beta 5  --BASE_DIR ~/MaxEntData4"
#--patterns 10 
# ARGS="--beta_max 6 --beta_min 6 --num_steps 10000 --rep 100000 --size 10 --num_beta 1  --BASE_DIR ~/MaxEntData4"

rm -rf ~/MaxEntData4
python generate_data_spin_model.py --seed 42    $ARGS
python calculate_Fig1a_v2.py                    $ARGS #--no_plot


