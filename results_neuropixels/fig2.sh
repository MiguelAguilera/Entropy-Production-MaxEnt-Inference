echo "Remember to make sure ep_data/* is clean"
#rm -rf ep_data/*
python calculate_Fig2a.py --BASE_DIR ~/Downloads/neuropixels --max_sessions 1 --lr 0.1 --obs 2 --patience 30 --tol 0  --seed 100  --order sorted --sizes 50 100 150 200 250
python display_Fig2a.py  --obs 2 --order sorted

