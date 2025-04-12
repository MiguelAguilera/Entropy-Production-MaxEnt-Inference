import argparse
import numpy as np
from matplotlib import pyplot as plt
import torch
from methods_EP_parallel import *
import os
from pathlib import Path
import h5py
import hdf5plugin

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
parser.add_argument("--rep", type=int, default="10",
                    help="Repetitions of neuron sampling for EP estimation (default: 10).")
parser.add_argument("--bin_size", type=float, default="0.01",
                    help="Bin size for neural spike disretization (default: 10).")
parser.add_argument("--order", type=str, default="random",
                    choices=["random", "sorted"],
                    help="Ordering of neurons: random or sorted by activity (default: random).")
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

        spike_sum = np.sum(S)

        dS = S[:, 1:] != S[:, :-1]
        num_changes = np.sum(dS, axis=0)
        one_spin_changes = np.mean(num_changes == 1)
        multi_spin_changes = np.mean(num_changes >= 2)

        T = S.shape[1]
        S_t = torch.from_numpy(S[:, 1:] * 2. - 1.)
        S1_t = torch.from_numpy(S[:, :-1] * 2. - 1.)

        rep = S_t.shape[1]

        if L2 == 'lin.1':
            lambda_ = 0.1 / N
        elif L2 == 'lin1':
            lambda_ = 1 / N
        else:
            lambda_ = 0

        print(f"  [Info] Estimating EP (repetitions: {rep})...")
        EP_maxent = get_torch(S_t, S1_t, mode=2, tol_per_param=1E-6, lambda_=lambda_)

        EP[n] = EP_maxent
        R[n] = spike_sum
        print(f"  [Result] EP: {EP_maxent:.5f} | Expected sum of spikes: {spike_sum:.5f}")

    save_path = f'data/neuropixels/neuropixels_{mode}_{order}_{session_type}_id_{session_id}_binsize_{bin_size}_L2_{L2}_rep_{r}.npz'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, R=R, EP=EP, sizes=sizes)
    print(f"\n[Saved] Results stored in: {save_path}")
    print("-" * 60)


# Run the pipeline
for r in range(rep):
    for session_id in range(102):
        for session_type in types:
            print(f"\n--- Running estimation for session {session_id} | Type: {session_type} | Repetition {r} ---")
            calc(sizes, session_type, session_id, r)
