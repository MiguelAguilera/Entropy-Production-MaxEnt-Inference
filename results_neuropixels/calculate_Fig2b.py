import argparse, sys, os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path
import numpy as np
import torch
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

sys.path.insert(0, '..')

import ep_estimators_bak as ep_estimators
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
parser.add_argument("--seed", type=int, default=0,
                    help="Observable (default: 0).")
parser.add_argument("--tol", type=float, default=1e-6)
parser.add_argument("--lr", type=float, default=1, help="Learning rate (default: 1)")
parser.add_argument("--lr_scale", type=str, choices=["none", "N", "sqrtN"], default="N",
                    help="Scale the learning rate by 'N', 'sqrtN', or use it as-is with 'none' (default: N).")
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience (default: 10)")
parser.add_argument("--overwrite", action="store_true", default=False,
                    help="Overwrite saved results (default: False).")
                    
args = parser.parse_args()


# ========================
# Plotting Configuration
# ========================

plt.rc('text', usetex=True)
font = {'size': 14, 'family': 'serif', 'serif': ['latin modern roman']}
plt.rc('font', **font)
plt.rc('legend', fontsize=12)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')


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

mode="visual"
order="sorted"

SAVE_DATA_DIR = Path("ep_fit_results")
SAVE_DATA_DIR.mkdir(exist_ok=True)

result_fname = SAVE_DATA_DIR / f'neuropixels_{mode}_{order}_binsize_{args.bin_size}_obs_{args.obs}_BB_lr_{args.lr}_lr-scale_{args.lr_scale}.h5'


if result_fname.exists() and not args.overwrite:
    print(f"[Skipping] Loading previously saved results from: {result_fname}")
    with h5py.File(result_fname, "r") as f:
        EP_maxent = f["EP"][()]
        theta_np = f["theta"][:]
    print(f"\nEP test: {EP_maxent:.5f}")
else:
    # --- Fit MaxEnt model ---
    
    if args.obs==1:
        data = observables.CrossCorrelations1(S_t, S1_t)
    elif args.obs==2:
        data = observables.CrossCorrelations2(S_t, S1_t)
    else:
        exit()
    trn, val, tst = data.split_train_val_test(val_fraction=0.2, test_fraction=0.1)
        
    torch.manual_seed(args.seed)
    print("→ Torch seed {args.seed}  set for CPU.")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        print("→ Torch seed {args.seed} set for CUDA.")
    trn, val, tst = data.split_train_val_test(val_fraction=0.2, test_fraction=0.1)

    if args.lr_scale == "none":
        lr_scaled = args.lr
    elif args.lr_scale == "N":
        lr_scaled = args.lr / args.N
    elif args.lr_scale == "sqrtN":
        lr_scaled = args.lr / np.sqrt(args.N)
    else:
        raise ValueError(f"Unknown lr_scale value: {args.lr_scale}")
        

    print(f"Training MaxEnt model (N={args.N}, tol={args.tol})...")
    optimizer_kwargs={}
    optimizer_kwargs['lr']=lr
    optimizer_kwargs['patience']=args.patience
    optimizer_kwargs['tol']=tol
    EP_maxent, theta = ep_estimators.get_EP_Estimate(trn, validation=val, test=tst,optimizer='GradientDescentBB', optimizer_kwargs=optimizer_kwargs)
#    EP_maxent, theta, EP_maxent_trn = ep_estimators.get_EP_GradientAscent(
#        data=trn,
#        validation_data=val,
#        test_data=tst,
#        lr=lr_scaled,
#        tol=args.tol,
#        use_Adam=args.use_Adam,
#        use_BB=args.use_BB,
#        patience=args.patience,
#        verbose=1,
#        beta1=args.Adam_args[0],
#        beta2=args.Adam_args[1],
#        eps=args.Adam_args[2]
#    )
    theta_np = theta.cpu().numpy()
    
    print(f"\nEP test: {EP_maxent:.5f} | EP trn: {EP_maxent_trn:.5f}")
    
    # Save results
    with h5py.File(result_fname, "w") as f:
        f.create_dataset("EP", data=EP_maxent)
        f.create_dataset("theta", data=theta_np)
    print(f"[Saved] Results to: {result_fname}")
    
N = args.N
if args.obs == 1:
    # θ is a vector of upper-triangular entries, reconstruct antisymmetric matrix
    th = np.zeros((N, N), dtype=theta_np.dtype)
    triu_indices = np.triu_indices(N, k=1)
    th[triu_indices] = theta_np
    th = th - th.T  # make antisymmetric
elif args.obs == 2:
    # θ is full matrix flattened or already N x N
    th = theta_np.reshape(N, N)

else:
    raise ValueError(f"Unknown obs value: {args.obs}")

# --- Cluster and sort θ ---
th_abs = np.abs(th)
th_abs=(th_abs + th_abs.T)/2
th_abs[range(N),range(N)]=0
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
    linthresh=0.02, linscale=0.10,
    vmin=-np.max(np.abs(th)), vmax=np.max(np.abs(th))
)

plt.figure(figsize=(5, 4))
plt.imshow(th, cmap='seismic', aspect='equal', interpolation='nearest', norm=norm)
plt.text(-0.28, 0.5, r'$\theta_{ij}^*$', fontsize=20, 
         va='center', ha='center', rotation=0, transform=plt.gca().transAxes)

for idx in area_start[1:]:
    plt.axhline(idx - 0.5, color='k', linewidth=1, linestyle=':')
    plt.axvline(idx - 0.5, color='k', linewidth=1, linestyle=':')

plt.xticks(area_centers, area_names, rotation=90)
plt.yticks(area_centers, area_names)
plt.tick_params(axis='both', which='major', labelsize=14)

cbar = plt.colorbar(shrink=0.85)
cbar.set_ticks([-0.06, -0.03,-0.015, 0, 0.015, 0.03, 0.06])
cbar.set_ticklabels([r'$-0.06$', r'$-0.03$', r'$-0.015$', r'$0$', r'$0.015$', r'$0.03$', r'$0.06$'])
cbar.ax.tick_params(length=6, width=1.5)
cbar.ax.minorticks_off()

plt.tight_layout()
plt.savefig('img/Fig2b.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

