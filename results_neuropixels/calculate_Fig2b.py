import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import h5py
import hdf5plugin  # Enables compressed HDF5 support
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# Add parent directory for custom module imports
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
    parser = argparse.ArgumentParser(description="Estimate EP with MaxEnt parameters and plot θ for Neuropixels.")

    parser.add_argument("--BASE_DIR", type=str, default="~/Neuropixels")
    parser.add_argument("--session_id", type=int, default=8)
    parser.add_argument("--session_type", type=str, choices=["active", "passive", "gabor"], default="active")
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--bin_size", type=float, default=0.01)
    parser.add_argument("--obs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--lr_scale", type=str, choices=["none", "N", "sqrtN"], default="N")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()

    # -------------------------------
    # Global Setup
    # -------------------------------
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    torch.set_grad_enabled(False)

    BASE_DIR = Path(args.BASE_DIR).expanduser()
    SAVE_DATA_DIR = Path("ep_fit_results")
    SAVE_DATA_DIR.mkdir(exist_ok=True)

    filename = BASE_DIR / f"data_binsize_{args.bin_size}_session_{args.session_id}.h5"
    result_fname = SAVE_DATA_DIR / f"neuropixels_visual_sorted_binsize_{args.bin_size}_obs_{args.obs}_BB_lr_{args.lr}_lr-scale_{args.lr_scale}.h5"

    print(f"Loading session {args.session_id} [{args.session_type}], N={args.N}")

    # -------------------------------
    # Load and Preprocess Data
    # -------------------------------
    with h5py.File(filename, 'r') as f:
        S = f[f'S_{args.session_type}'][:]
        areas = f['areas'][:].astype(str)

    # Filter visual neurons and sort
    visual_ix = np.where(np.char.startswith(areas, 'V'))[0]
    S, areas = S[visual_ix], areas[visual_ix]
    activity_order = np.argsort(-np.sum(S, axis=1))
    S, areas = S[activity_order[:args.N]], areas[activity_order[:args.N]]
    area_order = np.argsort(areas)
    S, areas = S[area_order], areas[area_order]

    # Convert to spin representation
    device = (                        torch.device("mps") if torch.backends.mps.is_available() else
                torch.device("cuda") if torch.cuda.is_available() else
                torch.device("cpu")
             )
    S_t = torch.from_numpy(S[:, 1:].T).to(device).float() * 2 - 1
    S1_t = torch.from_numpy(S[:, :-1].T).to(device).float() * 2 - 1

    # -------------------------------
    # Estimate MaxEnt Model
    # -------------------------------
    if result_fname.exists() and not args.overwrite:
        print(f"[Skip] Using existing results: {result_fname}")
        with h5py.File(result_fname, "r") as f:
            EP_maxent = f["EP"][()]
            theta = f["theta"][:]
    else:
        data = observables.CrossCorrelations1(S_t, S1_t) if args.obs == 1 else observables.CrossCorrelations2(S_t, S1_t)
        trn, val, tst = data.split_train_val_test(val_fraction=0.2, test_fraction=0.1)

        # Seed for reproducibility
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # Learning rate scaling
        if args.lr_scale == "none":
            lr_scaled = args.lr
        elif args.lr_scale == "N":
            lr_scaled = args.lr / args.N
        elif args.lr_scale == "sqrtN":
            lr_scaled = args.lr / np.sqrt(args.N)
        else:
            raise ValueError(f"Unknown lr_scale: {args.lr_scale}")

        optimizer_kwargs = {"lr": lr_scaled, "patience": args.patience, "tol": args.tol}
        EP_maxent, theta = ep_estimators.get_EP_Estimate(trn, validation=val, test=tst,
                                                         optimizer="GradientDescentBB",
                                                         optimizer_kwargs=optimizer_kwargs)

        with h5py.File(result_fname, "w") as f:
            f.create_dataset("EP", data=EP_maxent)
            f.create_dataset("theta", data=theta)
        print(f"[Saved] {result_fname}")

    print(f"\nEP test: {EP_maxent:.5f}")

    # -------------------------------
    # Process Theta Matrix
    # -------------------------------
    N = args.N
    if args.obs == 1:
        th = np.zeros((N, N), dtype=theta.dtype)
        i, j = np.triu_indices(N, k=1)
        th[i, j] = theta
        th = th - th.T
    elif args.obs == 2:
        th = theta.reshape(N, N)
    else:
        raise ValueError(f"Unknown observable: {args.obs}")

    # -------------------------------
    # Cluster and Reorder
    # -------------------------------
    th_abs = (np.abs(th) + np.abs(th).T) / 2
    np.fill_diagonal(th_abs, 0)
    linkage_matrix = linkage(squareform(th_abs), method="average")
    sorted_indices = leaves_list(linkage_matrix)

    th_sorted = th[sorted_indices][:, sorted_indices]
    areas = areas[sorted_indices]
    final_order = np.argsort(areas)
    th = th_sorted[final_order][:, final_order]
    areas = areas[final_order]

    area_names, area_start = np.unique(areas, return_index=True)
    area_end = np.r_[area_start[1:], [len(areas)]]
    area_centers = (area_start + area_end) / 2

    # -------------------------------
    # Plot Theta Matrix
    # -------------------------------
    import seaborn as sns
    sns.set(style='white', font_scale=1.4)
    plt.rc('text', usetex=True)
    plt.rc('font', size=12, family='serif', serif=['latin modern roman'])
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm,newtxtext,newtxmath}')

    norm = mcolors.SymLogNorm(linthresh=0.02, linscale=0.1,
                               vmin=-np.max(np.abs(th)), vmax=np.max(np.abs(th)))

    plt.figure(figsize=(5, 4))
    plt.imshow(th, cmap='seismic', aspect='equal', interpolation='nearest', norm=norm)
    plt.text(-0.28, 0.5, r'$\theta_{ij}^*$', fontsize=20, va='center', ha='center',
             rotation=0, transform=plt.gca().transAxes)

    for idx in area_start[1:]:
        plt.axhline(idx - 0.5, color='k', linewidth=1, linestyle=':')
        plt.axvline(idx - 0.5, color='k', linewidth=1, linestyle=':')

    plt.xticks(area_centers, area_names, rotation=90)
    plt.yticks(area_centers, area_names)
    plt.tick_params(axis='both', which='major', labelsize=14)

    cbar = plt.colorbar(shrink=0.85)
    cbar.set_ticks([-0.08, -0.04, -0.02, 0, 0.02, 0.04, 0.08])
    cbar.set_ticklabels([r'$-0.08$', r'$-0.04$', r'$-0.02$', r'$0$', r'$0.02$', r'$0.04$', r'$0.08$'])
    cbar.ax.tick_params(length=6, width=1.5)
    cbar.ax.minorticks_off()

    plt.tight_layout()
    IMG_DIR = Path("img")
    IMG_DIR.mkdir(exist_ok=True)
    plt.savefig(IMG_DIR / "Fig2b.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()

