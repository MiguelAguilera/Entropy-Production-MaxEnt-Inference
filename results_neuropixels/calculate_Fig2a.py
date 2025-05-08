import sys, argparse, os
from pathlib import Path
import numpy as np
import time

import h5py
import hdf5plugin

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import torch

sys.path.insert(0, '..')
from methods_EP_parallel import *
import ep_estimators
import utils
utils.set_default_torch_device()


torch.set_grad_enabled(False)

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
parser.add_argument("--seed", type=int, default=None,
                    help="Random number seed.")
parser.add_argument("--rep", type=int, default="1",
                    help="Repetitions of neuron sampling for EP estimation (default: 1).")
parser.add_argument("--bin_size", type=float, default="0.01",
                    help="Bin size for neural spike disretization (default: 10).")
parser.add_argument("--order", type=str, default="random",
                    choices=["random", "sorted","sorted_desc"],
                    help="Ordering of neurons: random or sorted by activity (default: random).")
parser.add_argument("--algorithm", type=str, default="GD",
                    choices=["GD", "LBFGS"],
                    help="Inference algorithm: GD or LBFGS (default: GD).")
parser.add_argument("--no_Adam", dest="use_Adam", action="store_false",
                    help="Disable Adam optimizer (enabled by default).")
parser.add_argument("--obs", type=int, default=1,
                    help="Observable (default: 1).")
parser.set_defaults(args=True)
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience for the optimizer (default: 10).")
parser.add_argument("--lr", type=float, default=0.01,
                    help="Base learning rate (default: 0.01).")
parser.add_argument("--lr_scale", type=str, choices=["none", "N", "sqrtN"], default="N",
                    help="Scale the learning rate by 'N', 'sqrtN', or use it as-is with 'none' (default: sqrtN).")
parser.add_argument("--tol", type=float, default=1e-6,
                    help="Base tolerance for convergence (default: 1e-6).")
parser.add_argument("--tol_scale", type=str, choices=["none", "N", "sqrtN"], default="N",
                    help="Scale the tolerance by 'N', 'sqrtN', or use it as-is with 'none' (default: N).")
parser.add_argument("--sizes", nargs="+", type=int,
                    default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                    help="List of population sizes to test (default: [50, 100, ..., 500]).")
parser.add_argument("--Adam_args", nargs=3, type=float, default=[0.8, 0.99, 1e-6],
                    help="Adam optimizer parameters: beta1, beta2, epsilon (default: 0.6 0.95 1e-6)")

parser.add_argument("--min_session", default=0, type=int,
                    help="First sessions to compute (default: 0).")
parser.add_argument("--max_session", default=103, type=int,
                    help="Max sessions to compute (default: 103).")

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

if args.seed is not None:
    print(f"Setting random seed to {args.seed}")
    np.random.seed(args.seed)

    
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
    
    
    theta_prev=None
    N_prev=None
    for n, N in enumerate(sizes):
        print(f"\n> Processing system size {N} neurons")
        stime = time.time()
        
        if mode == 'visual':
            indices = np.where(np.char.startswith(areas, 'V'))[0]
        elif mode == 'nonvisual':
            indices = np.where(~np.char.startswith(areas, 'V'))[0]
        elif mode == 'all':
            indices = np.arange(S_total.shape[0])

        S = S_total[indices, :]
        N0 = S.shape[0]

        if order != 'sorted':
            min_spikes = 10000
            print(f"!!! with order={order}, we restrict attention only to  neurons with >= {min_spikes} spikes")
            good_ixs = np.sum(S, axis=1)>=min_spikes
            S = S[good_ixs, :]
            N0 = S.shape[0]

        if N0 < N:
            print(f"[Info] Skipping size {N}: only {N0} units available after filtering.")
            EP[n:] = None
            break

        if order == 'random':
            inds = np.random.choice(N0, N, replace=False)
        elif order == 'sorted':
            inds = np.argsort(-np.sum(S, axis=1))[:N]
        elif order == 'sorted_desc':
            inds = np.argsort(np.sum(S, axis=1))[:N]
        S = S[inds, :].astype(DTYPE)

        # dS = S[:, 1:] != S[:, :-1]
        #num_changes = np.sum(dS, axis=0)
        #one_spin_changes = np.mean(num_changes == 1)
        #multi_spin_changes = np.mean(num_changes >= 2)
        S_t = torch.from_numpy(S[:, 1:].T * 2. - 1.).to(torch.get_default_device())
        S1_t = torch.from_numpy(S[:, :-1].T * 2. - 1.).to(torch.get_default_device())

        
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
            

        diff_mask = (S_t != S1_t).any(dim=1)
        print(f"  [Info] Estimating EP (nsamples: {nsamples}, lr: {lr:4f}, patience: {args.patience},"+
              f" % with transition: {diff_mask.float().mean().item():1f})...")

        if args.obs==1:
            data = ep_estimators.RawDataset(S_t, S1_t)
        elif args.obs==2:
            data = ep_estimators.RawDataset2(S_t, S1_t)
        else:
            exit()

        trn, tst = data.split_train_test(holdout_fraction=0.5, holdout_shuffle=True)
        spike_avg = (tst.X0+1).mean()*N/2 # number of spikes in test set

        theta_init=torch.zeros(N*(N-1)//2).to(torch.get_default_device())
        if theta_prev is not None and order != 'sorted':
            theta_init[:N_prev]=theta_prev
            N_prev=N
        start_time = time.time()
        EP_maxent_full,theta,EP_maxent_tst = ep_estimators.get_EP_GradientAscent(data=trn, holdout_data=tst, 
                                                lr=lr, tol=tol, use_Adam=args.use_Adam, patience=args.patience, 
                                                verbose=1, eps=1e-6,beta1=0.6, beta2=0.95#,report_every=10, 
                                                )
        
        del S_t, S1_t, data, trn, tst, theta  # free up memory explicitly
        utils.empty_torch_cache()

        EP[n] = EP_maxent_tst
        R[n] = spike_avg
        print(f"  [Result took {time.time()-stime:3f}] EP tst/full: {EP_maxent_tst:.5f} {EP_maxent_full:.5f} | R: {R[n]:.5f} | EP tst/R: {EP[n]/R[n]:.5f}")

    SAVE_DATA_DIR = 'ep_data'
    if args.use_Adam:
        adam_str = f'beta1_{args.Adam_args[0]}_beta2_{args.Adam_args[1]}_eps_{args.Adam_args[2]}'
        save_path = f'{SAVE_DATA_DIR}/neuropixels_{mode}_{order}_binsize_{bin_size}_obs_{args.obs}_Adam_lr_{args.lr}_lr-scale_{args.lr_scale}_args_{adam_str}.h5'
    else:
        save_path = f'{SAVE_DATA_DIR}/neuropixels_{mode}_{order}_binsize_{bin_size}_obs_{args.obs}_lr_{args.lr}_lr-scale_{args.lr_scale}.h5'

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
print(f'Doing {rep} repetitions')
for r in range(rep):
    for session_id in range(args.min_session, args.max_session):
        for session_type in types:
            print(f"\n--- Running estimation for session {session_id} | Type: {session_type} | Repetition {r} ---")
            calc(sizes, session_type, session_id, r)
