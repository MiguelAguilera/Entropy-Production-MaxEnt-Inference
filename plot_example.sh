#!/bin/sh
# 
ARGS="--patterns 10 --beta_min 1 --beta_max 4 --size 1000 --num_beta 5  --BASE_DIR ~/MaxEntData1000"
ARGS="--patterns 6 --beta_min 1 --beta_max 4 --rep 1000000 --size 200 --num_beta 5  --BASE_DIR ~/MaxEntData200"

ARGS="--patterns 6 --beta_min 0.01 --beta_max 4 --size 1000 --num_beta 10  --BASE_DIR ~/MaxEntData1000v2"
ARGS="--patterns 6 --beta_min 0.01 --beta_max 4 --size 400 --num_beta 10  --BASE_DIR ~/MaxEntData400"

# LAST ONE!!!!
ARGS="--patterns 6 --beta_min 0 --beta_max 3 --size 1000 --num_beta 25  --BASE_DIR ~/MaxEntData1000v3"

ARGS="--beta_min 2 --beta_max 4 --size 30 --num_beta 10  --BASE_DIR ~/MaxEntData30v2"
ARGS="--beta_min 0.01 --beta_max 4 --size 50 --num_beta 10  --BASE_DIR ~/MaxEntData50"

ARGS="--beta_min 0.01 --beta_max 2 --size 100 --num_beta 10  --BASE_DIR ~/MaxEntData100"

#ARGS="--rep 100000 --beta_min 0 --beta_max 4 --size 30 --num_beta 10  --BASE_DIR ~/MaxEntData30v5"
#ARGS="--rep 100000 --beta_min 4.5 --beta_max 8 --size 30 --num_beta 10  --BASE_DIR ~/MaxEntData30v6"
ARGS="--beta_min 5 --J0 0 --beta_max 5 --size 30 --num_beta 1  --BASE_DIR ~/MaxEntData30v10"

#ARGS="--beta_min 0 --J0 0 --beta_max 5 --size 30 --num_beta 10  --BASE_DIR ~/MaxEntData30v8_j0"
ARGS="--beta_min 0 --J0 0 --beta_max 5 --size 50 --num_beta 10  --BASE_DIR ~/MaxEntData30v7_j0"

ARGS="--beta_min 0 --rep 1000 --J0 0 --beta_max 5 --size 50 --num_beta 10  --BASE_DIR ~/MaxEntDataTest"

# ARGS="--beta_min 1 --J0 0 --beta_max 5 --size 1000 --num_beta 25  --BASE_DIR ~/MaxEntData1000diluted"

#ARGS="--beta_min 0  --beta_max 5 --size 30 --num_beta 10  --BASE_DIR ~/MaxEntData30v7"

# rm -rf ~/MaxEntData40v2
#ARGS="--patterns 6 --beta_min 0 --beta_max 3 --size 1000 --num_beta 1  --BASE_DIR ~/MaxEntData1000v11"
#ARGS="--patterns 100 --beta_min 1 --rep 1000000 --beta_max 4 --size 10 --num_beta 5  --BASE_DIR ~/MaxEntData10"
#python generate_data_spin_model.py  --seed 42    $ARGS

#--patterns 10 
# ARGS="--patterns 10 --beta_max 3.5 --beta_min 1 --num_steps 10000 --rep 100000 --size 1000 --num_beta 5  --BASE_DIR ~/MaxEntData4" 
#rm -rf ~/MaxEntData4

python generate_data_spin_model.py --seed 42   --num_neighbors 5  $ARGS 
python calculate_Fig1a_v2.py                    $ARGS # --overwrite #--no_plot


