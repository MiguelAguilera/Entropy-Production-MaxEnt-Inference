#!/bin/sh
# 
ARGS="--patterns 10 --beta_min 1 --beta_max 4 --size 1000 --num_beta 5  --BASE_DIR ~/MaxEntData1000"
ARGS="--patterns 6 --beta_min 1 --beta_max 4 --rep 1000000 --size 200 --num_beta 5  --BASE_DIR ~/MaxEntData200"

ARGS="--patterns 6 --beta_min 0.01 --beta_max 4 --size 1000 --num_beta 10  --BASE_DIR ~/MaxEntData1000v2"

#ARGS="--patterns 100 --beta_min 1 --rep 1000000 --beta_max 4 --size 10 --num_beta 5  --BASE_DIR ~/MaxEntData10"
#python generate_data_spin_model.py  --seed 42    $ARGS

#--patterns 10 
# ARGS="--patterns 10 --beta_max 3.5 --beta_min 1 --num_steps 10000 --rep 100000 --size 1000 --num_beta 5  --BASE_DIR ~/MaxEntData4" 
#rm -rf ~/MaxEntData4

#python generate_data_spin_model.py --seed 42    $ARGS
python calculate_Fig1a_v2.py                    $ARGS #--no_plot


