import sys, argparse, os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import time

import h5py
import hdf5plugin


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import torch

sys.path.insert(0, '..')
from methods_EP_parallel import *
import ep_estimators

parser = argparse.ArgumentParser(description="Estimate EP for Neuropixels data.")

parser.add_argument("--BASE_DIR", type=str, default="~/Neuropixels",
                    help="Base directory to store the data (default: '~/Neuropixels').")
parser.add_argument("--types", nargs="+", default=["active", "passive", "gabor"],
                    help="List of session types to include (default: active passive gabor).")
parser.add_argument("--mode", type=str, default="visual",
                    choices=["visual", "nonvisual", "all"],
                    help="Brain area mode to filter neurons (default: visual).")
parser.add_argument("--L2", type=str, default="0",
                    help="L2 regularization type: 0, lin1, lin.1 (default: 0).")
parser.add_argument("--rep", type=int, default="1",
                    help="Repetitions of neuron sampling for EP estimation (default: 10).")
parser.add_argument("--bin_size", type=float, default="0.01",
                    help="Bin size for neural spike disretization (default: 10).")
parser.add_argument("--order", type=str, default="random",
                    choices=["random", "sorted"],
                    help="Ordering of neurons: random or sorted by activity (default: random).")
parser.add_argument("--no_Adam", dest="use_Adam", action="store_false",
                    help="Disable Adam optimizer (enabled by default).")
parser.add_argument("--obs", type=float, default=1,
                    help="Observable (default: 1).")
parser.set_defaults(args=True)
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience for the optimizer (default: 10).")
parser.add_argument("--lr", type=float, default=0.01,
                    help="Base learning rate (default: 0.01).")
parser.add_argument("--lr_scale", type=str, choices=["none", "N", "sqrtN"], default="sqrtN",
                    help="Scale the learning rate by 'N', 'sqrtN', or use it as-is with 'none' (default: sqrtN).")
parser.add_argument("--tol", type=float, default=1e-6,
                    help="Base tolerance for convergence (default: 1e-6).")
parser.add_argument("--tol_scale", type=str, choices=["none", "N", "sqrtN"], default="N",
                    help="Scale the tolerance by 'N', 'sqrtN', or use it as-is with 'none' (default: N).")
parser.add_argument("--sizes", nargs="+", type=int,
                    default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                    help="List of population sizes to test (default: [50, 100, ..., 500]).")

args = parser.parse_args()

# --- Constants and Configuration ---
BASE_DIR = Path(args.BASE_DIR).expanduser()

DTYPE = 'float32'
bin_size = args.bin_size

types = ['active', 'passive', 'gabor']      # Conditions
mode = args.mode                            # Neural areas considered
L2 = args.L2                                # L2 regularization term
order = args.order                          # selected neurons

rep=args.rep                    # Repetitions of each EP estimation for different neuron samplings
sizes =args.sizes       # Sizes estimated



    
def calc(sizes, session_type, session_id, r):
    print()
    print("=" * 60)
    print(f"Starting EP estimation for session {session_id} [{session_type}] | Repetition {r}")
    print("=" * 60)

    empty_array = (np.array([None]), np.array([None]))

    filename = BASE_DIR / f'data_binsize_{bin_size}_session_{session_id}.h5'
    print(f"Loading data from: {filename}")

    try:
        with h5py.File(filename, 'r') as f:
            if session_type == 'active':
                S_total = f['S_active'][:]
            elif session_type == 'passive':
                S_total = f['S_passive'][:]
            elif session_type == 'gabor':
                S_total = f['S_gabor'][:]
            areas = f['areas'][:].astype(str)
    except Exception as e:
        print(f"[Warning] Failed to load session {session_id}: {e}")
        return empty_array

    EP = np.zeros(len(sizes))
    R = np.zeros(len(sizes))
    
    

    for n, N in enumerate(sizes):
        print(f"\n> Processing system size {N} neurons")

        
        if mode == 'visual':
            indices = np.where(np.char.startswith(areas, 'V'))[0]
        elif mode == 'nonvisual':
            indices = np.where(~np.char.startswith(areas, 'V'))[0]
        elif mode == 'all':
            indices = np.arange(S_total.shape[0])

        S = S_total[indices, :]
        N0 = S.shape[0]
        if N0 < N:
            print(f"[Info] Skipping size {N}: only {N0} units available after filtering.")
            EP[n:] = None
            break

        if order == 'random':
            inds = np.random.choice(N0, N, replace=False)
        elif order == 'sorted':
            inds = np.argsort(-np.sum(S, axis=1))[:N]
        S = S[inds, :].astype(DTYPE)

        spike_sum = np.mean(S)*N

        dS = S[:, 1:] != S[:, :-1]
        num_changes = np.sum(dS, axis=0)
        one_spin_changes = np.mean(num_changes == 1)
        multi_spin_changes = np.mean(num_changes >= 2)

        T = S.shape[1]
        
        
        
        S_t = torch.from_numpy(S[:, 1:].T * 2. - 1.)
        S1_t = torch.from_numpy(S[:, :-1].T * 2. - 1.)
        nsamples = S_t.shape[0]

        if L2 == 'lin.1':
            lambda_ = 0.1 / N
        elif L2 == 'lin1':
            lambda_ = 1 / N
        else:
            lambda_ = 0
            
        if args.lr_scale == "none":
            lr = args.lr
        elif args.lr_scale == "N":
            lr = args.lr / N
        elif args.lr_scale == "sqrtN":
            lr = args.lr / N**0.5
        else:
            raise ValueError(f"Unknown lr_scale value: {args.lr_scale}")
            
        if args.tol_scale == "none":
            tol = args.tol
        elif args.tol_scale == "N":
            tol = args.tol / N
        elif args.tol_scale == "sqrtN":
            tol = args.tol / N**0.5
        else:
            raise ValueError(f"Unknown tol_scale value: {args.tol_scale}")
            

        print(f"  [Info] Estimating EP (nsamples: {nsamples})...")
        
#        
#        start_time = time.time()
#        EP_maxent = get_torch(S_t.T, S1_t.T, mode=2, tol_per_param=1E-6, lambda_=lambda_)
#        print(f"Time for Gradient Ascent2: {time.time() - start_time:.4f} seconds")
#        print(f"  [Result] EP: {EP_maxent:.5f} | Expected sum of spikes R: {spike_sum:.5f} | EP/R: {EP_maxent/spike_sum:.5f}")
#        print()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        S_t = torch.from_numpy(S[:, 1:].T * 2. - 1.).to(device)
        S1_t = torch.from_numpy(S[:, :-1].T * 2. - 1.).to(device)


        if args.obs==1:
            data = ep_estimators.RawDataset(S_t, S1_t)
        elif args.obs==2:
            data = ep_estimators.RawDataset2(S_t, S1_t)
        else:
            exit()
        ep_est = ep_estimators.EPEstimators(data)

        start_time = time.time()
        lr=1e-2
        max_iter = 5000
        print(lr)
        
        EP_maxent,theta,_ = ep_est.get_EP_GradientAscent(lr = lr, 
                                                         holdout=True, 
                                                         tol=tol, 
                                                         use_Adam=args.use_Adam, 
                                                         patience=args.patience, 
                                                         verbose=2,
                                                         max_iter=max_iter)
#        print(f"Time for Gradient Ascent: {time.time() - start_time:.4f} seconds")
        
        if device.type == "cuda":
            del S_t, S1_t, data, ep_est, theta  # free up memory explicitly
            torch.cuda.empty_cache()
#        print(E, E2, EP_maxent)
#        exit()
        
#        ep_estimator = EPEstimators(S_t.T, S1_t.T)
#        ep_estimator.get_EP_GradientAscent()

        EP[n] = EP_maxent
        R[n] = spike_sum
        print(f"  [Result] EP: {EP_maxent:.5f} | Expected sum of spikes R: {spike_sum:.5f} | EP/R: {EP_maxent/spike_sum:.5f}")

    SAVE_DATA_DIR = 'ep_data'
    save_path = f'{SAVE_DATA_DIR}/neuropixels_{mode}_{order}_binsize_{bin_size}_obs_{args.obs}_Adam_{args.use_Adam}.h5'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(save_path, 'a') as f:
        group_path = f"{session_type}/{session_id}/rep_{r}"
        if group_path in f:
            del f[group_path]  # Overwrite the group if it exists
        grp = f.create_group(group_path)
        grp.create_dataset('EP', data=EP)
        grp.create_dataset('R', data=R)
        grp.create_dataset('sizes', data=sizes)
    print(f"\n[Saved] Results stored in: {save_path}")
    print("-" * 60)


# Run the pipeline
for r in range(rep):
    for session_id in range(103):
        for session_type in types:
            print(f"\n--- Running estimation for session {session_id} | Type: {session_type} | Repetition {r} ---")
            calc(sizes, session_type, session_id, r)
