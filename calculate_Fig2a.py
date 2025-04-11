import numpy as np
from matplotlib import pyplot as plt
from numba import njit, cuda
import time
import torch
from torchmin import minimize
from methods_EP_parallel import *
import os
from pathlib import Path
import h5py

parser = argparse.ArgumentParser(description="Estimate EP for neuropixels data.")
parser.add_argument("--BASE_DIR", type=str, default="~/Neuropixels",
                    help="Base directory to store the data (default: '~/Neuropixels').")
args = parser.parse_args()


# --- Constants and Configuration ---
BASE_DIR = Path(args.BASE_DIR).expanduser()

DTYPE = 'float32'
bin_size = 0.01

types = ['active', 'passive', 'gabor']
mode = 'visual'
L2 = '0'
order = 'random'

max_size = 200
sizes = [int(i * (max_size / 10)) for i in range(1, 11)]

# Main function to run calculations for a given system size
def calc(sizes, session_type, session_id):
    print()
    start_time = time.time()
    empty_array = (np.array([None]), np.array([None]))

    filename = BASE_DIR / f'data_binsize_{bin_size}_session_{session_id}.h5'
    print(filename)

    try:
        with h5py.File(filename, 'r') as f:
            if session_type == 'active':
                S = f['S_active'][:]
            elif session_type == 'passive':
                S = f['S_passive'][:]
            elif session_type == 'gabor':
                S = f['S_gabor'][:]
            areas = f['areas'][:].astype(str)
    except Exception as e:
        print(f"Failed to load session {session_id}: {e}")
        return empty_array

    EPR = np.zeros(len(sizes))
    fr = np.zeros(len(sizes))

    for n, N in enumerate(sizes):
        print(f'** DOING SESSION {session_id}, SYSTEM SIZE {N} **', flush=True)

        indices = np.arange(S.shape[0])
        if mode == 'visual':
            indices = np.where(np.char.startswith(areas, 'V'))[0]
        elif mode == 'visual+':
            indices = np.where(np.char.startswith(areas, 'V') | np.isin(areas, ['APN', 'LGN', 'LGd']))[0]
        elif mode == 'nonvisual':
            indices = np.where(~np.char.startswith(areas, 'V'))[0]
        elif mode == 'midbrain':
            indices = np.where(np.isin(areas, ['SCig', 'MRN', 'MB', 'ZI']))[0]
        elif mode == 'attention':
            indices = np.where(np.isin(areas, ['SCig', 'ZI', 'MRN','POST','MB','MGm','MGd','SGN','TH','CA1','DG']))[0]

        S = S[indices, :]
        N0 = S.shape[0]
        if N0 < N:
            EPR[n:] = None
            break

        if order == 'random':
            inds = np.random.choice(N0, N, replace=False)
        elif order == 'sorted':
            inds = np.argsort(-np.sum(S, axis=1))[:N]
        S = S[inds, :].astype(DTYPE)

        firing_rate = np.mean(S)

        dS = S[:, 1:] != S[:, :-1]
        num_changes = np.sum(dS, axis=0)
        one_spin_changes = np.mean(num_changes == 1)
        multi_spin_changes = np.mean(num_changes >= 2)

        T = S.shape[1]
        S_t = torch.from_numpy(S[:, 1:] * 2. - 1.)
        S1_t = torch.from_numpy(S[:, :-1] * 2. - 1.)

        print(S_t.shape)
        rep = S_t.shape[1]

        if L2 == 'lin.1':
            lambda_ = 0.1 / N
        elif L2 == 'lin1':
            lambda_ = 1 / N
        else:
            lambda_ = 0

        start_time = time.perf_counter()
        sig_maxent2 = get_torch(S_t, S1_t, mode=2, tol_per_param=1E-6, lambda_=lambda_)
        time_maxent2 = time.perf_counter() - start_time

        EPR[n] = sig_maxent2
        fr[n] = firing_rate
        print('EP', sig_maxent2, 'R', firing_rate)

    save_path = f'data/neuropixels/neuropixels_{mode}_{order}_{session_type}_id_{session_id}_binsize_{bin_size}_L2_{L2}.npz'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, fr=fr, EPR=EPR, sizes=sizes)

# Run the pipeline
ind = 0
for session_id in range(103):
    for session_type in types:
        print('TYPE', session_type)
        calc(sizes, session_type, session_id)

