#!/bin/sh

rm -rf ~/MaxEntData2

python generate_data_spin_model.py --seed 42 --size 20 --num_beta 5  --BASE_DIR ~/MaxEntData2
python fig1a.py                              --size 20 --num_beta 5   --BASE_DIR ~/MaxEntData2


