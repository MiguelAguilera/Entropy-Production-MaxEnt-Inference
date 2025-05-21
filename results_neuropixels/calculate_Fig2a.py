import os
import sys
import argparse
import numpy as np
import torch
import h5py
import time
from pathlib import Path

import hdf5plugin  # noqa: F401, required for compressed HDF5

# Import custom modules
sys.path.insert(0, "..")
import ep_estimators
import observables
import utils

# -------------------------------
# Main Entry Point
# -------------------------------
if __name__ == "__main__":

    # -------------------------------
    # Argument Parsing
    # -------------------------------
    parser = argparse.ArgumentParser(description="Estimate EP for Neuropixels data.")

    parser.add_argument("--BASE_DIR", type=str, default="~/Neuropixels")
    parser.add_argument("--types", nargs="+", default=["active", "passive", "gabor"])
    parser.add_argument("--mode", type=str, choices=["visual", "nonvisual", "all"], default="visual")
    parser.add_argument("--L2", type=str, default="0")
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--bin_size", type=float, default=0.01)
    parser.add_argument("--order", type=str, choices=["random", "sorted", "sorted_desc"], default="random")
    parser.add_argument("--obs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--lr_scale", type=str, choices=["none", "N", "sqrtN"], default="N")
    parser.add_argument("--tol", type=float, default=0)
    parser.add_argument("--tol_scale", type=str, choices=["none", "N", "sqrtN"], default="N")
    parser.add_argument("--sizes", nargs="+", type=int, default=[50, 100, 150, 200, 250, 300])
    parser.add_argument("--min_session", type=int, default=0)
    parser.add_argument("--max_session", type=int, default=103)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    # -------------------------------
    # Global Setup
    # -------------------------------

    BASE_DIR = Path(args.BASE_DIR).expanduser()
    SAVE_DATA_DIR = "ep_data"
    os.makedirs(SAVE_DATA_DIR, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)

    # -------------------------------
    # EP Estimation Loop
    # -------------------------------
    def estimate_EP_for_session(session_type, session_id, r):
        print(f"\n>>> Session {session_id} | Type: {session_type} | Repetition {r}")

        filename = BASE_DIR / f"data_binsize_{args.bin_size}_session_{session_id}.h5"
        group_path = f"{session_type}/{session_id}/rep_{r}"
        save_path = f"{SAVE_DATA_DIR}/neuropixels_{args.mode}_{args.order}_binsize_{args.bin_size}_obs_{args.obs}_BB_lr_{args.lr}_lr-scale_{args.lr_scale}.h5"

        if os.path.exists(save_path):
            with h5py.File(save_path, 'r') as f:
                if group_path in f and not args.overwrite:
                    print("[Skip] Already exists.")
                    return

        try:
            with h5py.File(filename, 'r') as f:
                S_total = f[f"S_{session_type}"][:]
                areas = f['areas'][:].astype(str)
        except Exception as e:
            print(f"[Error] Failed to load session {session_id}: {e}")
            return

        EP = np.zeros(len(args.sizes))
        R = np.zeros(len(args.sizes))

        for i, N in enumerate(args.sizes):
            print(f"  - Size {N}", end="")
            if args.mode == 'visual':
                indices = np.where(np.char.startswith(areas, 'V'))[0]
            elif args.mode == 'nonvisual':
                indices = np.where(~np.char.startswith(areas, 'V'))[0]
            else:
                indices = np.arange(S_total.shape[0])

            S = S_total[indices, :]

            if args.order != 'sorted':
                min_spikes_ratio = 0.02
                mask = np.mean(S, axis=1) >= min_spikes_ratio
                S = S[mask, :]

            if S.shape[0] < N:
                print(f"    [Skip] Not enough neurons after filtering: {S.shape[0]} < {N}")
                EP[i:] = None
                break

            if args.order == 'random':
                inds = np.random.choice(S.shape[0], N, replace=False)
            elif args.order == 'sorted':
                inds = np.argsort(-np.sum(S, axis=1))[:N]
            else:
                inds = np.argsort(np.sum(S, axis=1))[:N]

            device = (                        torch.device("mps") if torch.backends.mps.is_available() else
                        torch.device("cuda") if torch.cuda.is_available() else
                        torch.device("cpu")
                     )
            S = S[inds, :]
            S_t = torch.from_numpy(S[:, 1:].T).to(device).float() * 2 - 1
            S1_t = torch.from_numpy(S[:, :-1].T).to(device).float() * 2 - 1

            data = observables.CrossCorrelations1(S_t, S1_t) if args.obs == 1 else observables.CrossCorrelations2(S_t, S1_t)
            trn, val, tst = data.split_train_val_test(val_fraction=0.2, test_fraction=0.1)

            spike_avg = (tst.X0 + 1).mean() * N / 2

            lr_scaled = args.lr / (N if args.lr_scale == "N" else N**0.5 if args.lr_scale == "sqrtN" else 1)
            tol = args.tol / (N if args.tol_scale == "N" else N**0.5 if args.tol_scale == "sqrtN" else 1)

            optimizer_kwargs = {"lr": lr_scaled, "patience": args.patience, "tol": tol}
            EP_val, theta = ep_estimators.get_EP_Estimate(trn, validation=val, test=tst, optimizer='GradientDescentBB', optimizer_kwargs=optimizer_kwargs)

            EP[i] = EP_val
            R[i] = spike_avg
            utils.empty_torch_cache()

            print(f" | EP: {EP_val:.5f} | R: {spike_avg:.5f} | EP/R: {EP_val/spike_avg:.5f}")

        with h5py.File(save_path, 'a') as f:
            if group_path in f:
                del f[group_path]
            grp = f.create_group(group_path)
            grp.create_dataset("EP", data=EP)
            grp.create_dataset("R", data=R)
            grp.create_dataset("sizes", data=args.sizes)
        print(f"[Saved] {save_path}")

    # -------------------------------
    # Run All Sessions
    # -------------------------------
    print(f"Running {args.rep} repetitions")
    for r in range(args.rep):
        for session_id in range(args.min_session, args.max_session):
            for session_type in args.types:
                estimate_EP_for_session(session_type, session_id, r)

