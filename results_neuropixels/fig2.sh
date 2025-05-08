echo "Remember to make sure ep_data/* is clean"
#rm -rf ep_data/*
ARGS="--obs 2  --order sorted"
#ARGS="--obs 2  --order random"

python calculate_Fig2a.py --BASE_DIR ~/Downloads/neuropixels --min_session 1 --max_session 10 --no_Adam --lr 0.25 --patience 30 --tol 0  --seed 100  $ARGS --sizes 50 100 150 200 250
python display_Fig2a.py   $ARGS

