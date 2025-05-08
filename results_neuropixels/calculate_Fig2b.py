import argparse, sys, os
from pathlib import Path
import numpy as np
import torch
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.insert(0, '..')

import ep_estimators
import utils
utils.set_default_torch_device()
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description="Estimate EP with MaxEnt parameters and plot θ for Neuropixels.")
parser.add_argument("--BASE_DIR", type=str, default="~/Neuropixels")
parser.add_argument("--session_id", type=int, default=8)
parser.add_argument("--session_type", type=str, choices=["active", "passive", "gabor"], default="active")
parser.add_argument("--N", type=int, default=200)
parser.add_argument("--bin_size", type=float, default=0.01)
parser.add_argument("--obs", type=int, default=1)
parser.add_argument("--tol", type=float, default=1e-6)
parser.add_argument("--no_Adam", dest="use_Adam", action="store_false",
                    help="Disable Adam optimizer (enabled by default).")
parser.set_defaults(use_Adam=True)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
parser.add_argument("--lr_scale", type=str, choices=["none", "N", "sqrtN"], default="N",
                    help="Scale the learning rate by 'N', 'sqrtN', or use it as-is with 'none' (default: N).")
parser.add_argument("--Adam_args", nargs=3, type=float, default=[0.9, 0.999, 1e-8],
                    help="Adam parameters: beta1 beta2 eps (default: 0.9 0.999 1e-8)")
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience (default: 10)")

args = parser.parse_args()

# --- Constants and Paths ---
BASE_DIR = Path(args.BASE_DIR).expanduser()
filename = BASE_DIR / f"data_binsize_{args.bin_size}_session_{args.session_id}.h5"
print(f"** Loading session {args.session_id} [{args.session_type}], N={args.N} **")

# --- Load and preprocess data ---
with h5py.File(filename, 'r') as f:
    key = f'S_{args.session_type}'
    S = f[key][:]
    areas = f['areas'][:].astype(str)

# Filter and sort neurons
visual_ix = np.where(np.char.startswith(areas, 'V'))[0]
S, areas = S[visual_ix], areas[visual_ix]

activity_order = np.argsort(-np.sum(S, axis=1))
S, areas = S[activity_order[:args.N]], areas[activity_order[:args.N]]

area_order = np.argsort(areas)
S, areas = S[area_order], areas[area_order]

S_t = torch.from_numpy(S[:, 1:].T * 2. - 1.).float().to(torch.get_default_device())
S1_t = torch.from_numpy(S[:, :-1].T * 2. - 1.).float().to(torch.get_default_device())

# --- Fit MaxEnt model ---
data = ep_estimators.RawDataset(S_t, S1_t) if args.obs == 1 else ep_estimators.RawDataset2(S_t, S1_t)
trn, tst = data.split_train_test(holdout_fraction=0.5, holdout_shuffle=True)

if args.lr_scale == "none":
    lr_scaled = args.lr
elif args.lr_scale == "N":
    lr_scaled = args.lr / args.N
elif args.lr_scale == "sqrtN":
    lr_scaled = args.lr / np.sqrt(args.N)
else:
    raise ValueError(f"Unknown lr_scale value: {args.lr_scale}")
    
    
print(f"Training MaxEnt model (N={args.N}, tol={args.tol})...")
EP_maxent_tst, theta, EP_maxent_full = ep_estimators.get_EP_GradientAscent(
    data=trn,
    holdout_data=tst,
    lr=lr_scaled,
    tol=args.tol,
    use_Adam=args.use_Adam,
    patience=args.patience,
    verbose=1,
    beta1=args.Adam_args[0],
    beta2=args.Adam_args[1],
    eps=args.Adam_args[2]
)
print(f"\nEP test: {EP_maxent_tst:.5f} | EP full: {EP_maxent_full:.5f}")

# --- Extract θ matrix ---
theta_np = theta.cpu().numpy()
N = args.N
th = np.zeros((N, N), dtype=theta_np.dtype)
triu_indices = np.triu_indices(N, k=1)
th[triu_indices] = theta_np
th = th - th.T  # Make antisymmetric

# --- Cluster and sort θ ---
th_abs = np.abs(th)
linkage_matrix = linkage(squareform(th_abs), method='average')
sorted_indices = leaves_list(linkage_matrix)

th_sorted = th[sorted_indices][:, sorted_indices]
areas = areas[sorted_indices]

final_order = np.argsort(areas)
th = th_sorted[final_order][:, final_order]
areas = areas[final_order]

area_names, area_start = np.unique(areas, return_index=True)
area_end = np.r_[area_start[1:], [len(areas)]]
area_centers = (area_start + area_end) / 2

# --- Plot θ matrix ---
norm = mcolors.SymLogNorm(
    linthresh=0.012, linscale=0.05,
    vmin=-np.max(np.abs(th)), vmax=np.max(np.abs(th))
)

plt.figure(figsize=(5, 4))
plt.imshow(th, cmap='bwr', aspect='equal', interpolation='nearest', norm=norm)
plt.text(-0.25, 0.5, r'$\theta_{ij}^*$', fontsize=18, 
         va='center', ha='center', rotation=0, transform=plt.gca().transAxes)

for idx in area_start[1:]:
    plt.axhline(idx - 0.5, color='k', linewidth=1, linestyle=':')
    plt.axvline(idx - 0.5, color='k', linewidth=1, linestyle=':')

plt.xticks(area_centers, area_names, rotation=90)
plt.yticks(area_centers, area_names)
plt.tick_params(axis='both', which='major', labelsize=14)

cbar = plt.colorbar(shrink=0.85)
cbar.set_ticks([-0.1, -0.05, -0.025, 0, 0.025, 0.05, 0.1])
cbar.set_ticklabels([r'$-0.1$', r'$-0.05$', r'$-0.025$', r'$0$', r'$0.025$', r'$0.05$', r'$0.1$'])
cbar.ax.tick_params(length=6, width=1.5)
cbar.ax.minorticks_off()

plt.tight_layout()
plt.savefig('img/Fig2b.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

