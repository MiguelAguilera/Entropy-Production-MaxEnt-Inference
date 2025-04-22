#!/bin/sh

ARGS="--patterns 10 --beta_max 10 --size 10 --num_beta 10  --BASE_DIR ~/MaxEntData2"
rm -rf ~/MaxEntData2
python generate_data_spin_model.py --seed 42 $ARGS
python calculate_Fig1a_v2.py                    $ARGS


