#!/bin/sh
# 
ARGS="--patterns 10 --beta_min 1 --beta_max 4 --N 1000 --num_beta 5  --BASE_DIR ~/MaxEntData1000"
ARGS="--patterns 6 --beta_min 1 --beta_max 4 --rep 1000000 --N 200 --num_beta 5  --BASE_DIR ~/MaxEntData200"

ARGS="--patterns 6 --beta_min 0.01 --beta_max 4 --N 1000 --num_beta 10  --BASE_DIR ~/MaxEntData1000v2"
ARGS="--patterns 6 --beta_min 0.01 --beta_max 4 --N 400 --num_beta 10  --BASE_DIR ~/MaxEntData400"

# LAST ONE!!!!
ARGS="--patterns 6 --beta_min 0 --beta_max 3 --N 1000 --num_beta 25  --BASE_DIR ~/MaxEntData1000v3"

ARGS="--beta_min 2 --beta_max 4 --N 30 --num_beta 10  --BASE_DIR ~/MaxEntData30v2"
ARGS="--beta_min 0.01 --beta_max 4 --N 50 --num_beta 10  --BASE_DIR ~/MaxEntData50"

ARGS="--beta_min 0.01 --beta_max 2 --N 100 --num_beta 10  --BASE_DIR ~/MaxEntData100"

#ARGS="--rep 100000 --beta_min 0 --beta_max 4 --N 30 --num_beta 10  --BASE_DIR ~/MaxEntData30v5"
#ARGS="--rep 100000 --beta_min 4.5 --beta_max 8 --N 30 --num_beta 10  --BASE_DIR ~/MaxEntData30v6"
ARGS="--beta_min 5 --J0 0 --beta_max 5 --N 30 --num_beta 1  --BASE_DIR ~/MaxEntData30v10"

#ARGS="--beta_min 0 --J0 0 --beta_max 5 --N 30 --num_beta 10  --BASE_DIR ~/MaxEntData30v8_j0"
ARGS="--beta_min 0 --J0 0 --beta_max 5 --N 50 --num_beta 10  --BASE_DIR ~/MaxEntData30v7_j0"

ARGS="--beta_min 0 --rep 1000 --J0 0 --beta_max 5 --N 50 --num_beta 10  --BASE_DIR ~/MaxEntDataTest"

ARGS="--beta_min 0 --J0 0 --beta_max 5 --N 1000 --num_beta 25  --BASE_DIR ~/MaxEntData1000diluted"

ARGS="--beta_min 0 --J0 0 --beta_max 4 --N 200 --num_beta 25  --BASE_DIR ~/MaxEntData200diluted"



#ARGS="--beta_min 3 --J0 0 --beta_max 5 --N 30 --num_beta 1  --BASE_DIR ~/MaxEntDataTest100"

#ARGS="--beta_min 3 --J0 0 --beta_max 4 --N 40 --num_beta 1  --BASE_DIR ~/MaxEntDataTest40"


#ARGS="--beta_min 0  --beta_max 5 --N 30 --num_beta 10  --BASE_DIR ~/MaxEntData30v7"

ARGS="--beta_min 0 --DJ 1 --J0 0 --beta_max 2 --N 1000 --rep 10000000 --num_beta 25  --BASE_DIR ~/MaxEntDataSteps"

ARGS="--beta_min 0 --rep 2000000 --DJ 1 --J0 0 --beta_max 6 --N 50 --num_beta 25  --BASE_DIR ~/MaxEntDataTest50v4"

#ARGS="--beta_min 0 --rep 20000 --DJ 1 --J0 0 --beta_max 6 --N 50 --num_beta 10  --BASE_DIR ~/MaxEntDataTest50v10"

#ARGS="--beta_min 0 --rep 200000 --DJ 1 --J0 0 --beta_max 3.5 --N 1000 --num_beta 5  --BASE_DIR ~/MaxEntDataTest1000v3"
#ARGSEP="--nograd --overwrite"
# ARGS="--beta_min 3.5 --J0 0 --DJ 1 --beta_max 4 --N 500 --num_beta 1  --BASE_DIR ~/MaxEntDataTest500dj1"

#ARGS="--beta_min 0 --J0 0 --DJ 1 --beta_max 4 --N 500 --num_beta 10  --BASE_DIR ~/MaxEntData500dil"

#ARGS="--beta_min 0 --J0 0 --DJ 1 --beta_max 4 --N 1000 --num_beta 10  --BASE_DIR ~/MaxEntData1000_1m"

#ARGS="--beta_min 0 --J0 0 --DJ 1 --beta_max 4 --N 1000 --num_beta 10  --BASE_DIR ~/MaxEntData1000_10k"
#rm -rf ~/MaxEntData500dilS
# rm -rf ~/MaxEntData40v2
#ARGS="--patterns 6 --beta_min 0 --beta_max 3 --N 1000 --num_beta 1  --BASE_DIR ~/MaxEntData1000v11"
#ARGS="--patterns 100 --beta_min 1 --rep 1000000 --beta_max 4 --N 10 --num_beta 5  --BASE_DIR ~/MaxEntData10"
#python generate_data_spin_model.py  --seed 42    $ARGS

ARGS="--beta_min 0 --rep 1000 --DJ 1 --J0 0 --beta_max 4 --N 100 --num_beta 10  --BASE_DIR ~/MaxEntData100-v1k"
ARGS="--beta_min 0 --rep 100000 --DJ 1 --J0 0 --beta_max 4 --N 50 --num_beta 10  --BASE_DIR ~/MaxEntData50-v100k"
#ARGS="--beta_min 0 --rep 10000 --DJ 1 --J0 0 --beta_max 4 --N 50 --num_beta 10  --BASE_DIR ~/MaxEntData50-v10k"
#ARGS="--beta_min 0 --rep 2000000 --DJ 1 --J0 0 --beta_max 4 --N 50 --num_beta 10  --BASE_DIR ~/MaxEntDataTest50"
#ARGS="--beta_min 0 --J0 0 --beta_max 5 --N 300 --num_beta 25  --BASE_DIR ~/MaxEntData300diluted"
ARGS="--beta_min 0 --rep 200000 --DJ 1 --J0 0 --beta_max 6 --N 50 --num_beta 10  --BASE_DIR ~/MaxEntDataTest50v6"

#ARGS="--beta_min 0 --J0 0 --DJ 1 --beta_max 4 --N 1000 --num_beta 10  --BASE_DIR ~/MaxEntData1000dil"

#ARGS="--beta_min 0 --rep 10000 --DJ 1 --J0 0 --beta_max 4 --N 100 --num_beta 10  --BASE_DIR ~/MaxEntData100-v10k"
#ARGS="--beta_min 0 --rep 100000 --DJ 1 --J0 0 --beta_max 4 --N 100 --num_beta 10  --BASE_DIR ~/MaxEntData100-v100k"

#--patterns 10 
# ARGS="--patterns 10 --beta_max 3.5 --beta_min 1 --num_steps 10000 --rep 100000 --N 1000 --num_beta 5  --BASE_DIR ~/MaxEntData4" 
#rm -rf ~/MaxEntData4
ARGS="--beta_min 0 --J0 0 --DJ 1 --beta_max 4 --N 500 --num_beta 10  --BASE_DIR ~/MaxEntData500dil"

ARGS="--beta_min 0 --rep 10000 --DJ 1 --J0 0 --beta_max 4 --N 100 --num_beta 10  --BASE_DIR ~/MaxEntData100-v10k"
ARGS="--beta_min 0 --rep 100000 --DJ 1 --J0 0 --beta_max 4 --N 20 --num_beta 10  --BASE_DIR ~/MaxEntData20"

ARGS="--beta_min 0 --rep 100000 --DJ 1 --J0 0 --beta_max 4 --N 100 --num_beta 5  --BASE_DIR ~/MaxEntData100"

ARGS="--beta_min 0 --beta_max 4 --DJ 1 --J0 0 --N 100 --num_beta 1  --BASE_DIR ~/MaxEntData100"

ARGS="--rep 2000000 --beta_min 0 --beta_max 4 --DJ 1 --J0 0 --N 1000 --num_beta 5  --BASE_DIR ~/MaxEntData1000_2M"
ARGS=" --beta_min 0 --beta_max 4 --DJ 1 --J0 0 --N 1000 --num_beta 5  --BASE_DIR ~/MaxEntData1000"

#python generate_data_spin_model.py --seed 42   --num_neighbors 6  $ARGS 
python scaling_fig.py                    $ARGS  $ARGSEP --overwrite #--nograd --overwrite  #--nograd   --overwrite 
 
# python calculate_Fig1a_v2.py                    $ARGS  $ARGSEP  --overwrite  --nograd  #--nograd   --overwrite 

