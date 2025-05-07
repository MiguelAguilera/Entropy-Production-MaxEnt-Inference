import argparse, sys, os
import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import matplotlib.colors as mcolors
import h5py
import hdf5plugin

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import torch


sys.path.insert(0, '..')
from methods_EP_parallel import *
import ep_estimators, utils
utils.set_default_torch_device()
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description="Estimate EP and MaxEnt parameters for a neuropixels dataset.")
parser.add_argument("--BASE_DIR", type=str, default="~/Neuropixels",
                    help="Base directory to store the data (default: '~/Neuropixels').")
parser.add_argument("--obs", type=int, default=1,
                    help="Observable (default: 1).")
parser.add_argument("--fancy_color_scale", dest="fancy_color_scale", action="store_true",
                    help="Fancy color rescaling.")
args = parser.parse_args()


# --- Constants and Configuration ---
BASE_DIR = Path(args.BASE_DIR).expanduser()
DTYPE = 'float32'
tol_per_param = 1e-6
bin_size = 0.01
session_id = 0 # 8
session_type = 'active'
N = 200  # Number of neurons






# Configure LaTeX rendering in matplotlib
plt.rc('text', usetex=True)
plt.rc('font', size=18, family='serif', serif=['latin modern roman'])
plt.rc('legend', fontsize=18)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')

print(f"** DOING SYSTEM SIZE {N} of type {session_type} with session ID {session_id} **", flush=True)

# --- Load and preprocess data ---
filename = BASE_DIR / f"data_binsize_{bin_size}_session_{session_id}.h5"
print(f'Loading data from {filename}, session_type={session_type}')

with h5py.File(filename, 'r') as f:
    if session_type == 'active':
        S = f['S_active'][:]
    elif session_type == 'passive':
        S = f['S_passive'][:]
    elif session_type == 'gabor':
        S = f['S_gabor'][:]
    areas = f['areas'][:].astype(str)  # decode bytes to UTF-8

# Select only visual areas (e.g., V1, V2, etc.)
indices = np.where(np.char.startswith(areas, 'V'))[0]
S = S[indices, :]
areas = areas[indices]

# Sort neurons by overall activity
inds = np.argsort(-np.sum(S, axis=1))
S = S[inds[:N], :].astype(DTYPE) * 2. - 1.
areas = areas[inds[:N]]

# Sort neurons by area name
inds2 = np.argsort(areas)
S = S[inds2, :]
areas = areas[inds2]

# --- Fit MaxEnt model ---
S_t = torch.from_numpy(S[:, 1:]).T
S1_t = torch.from_numpy(S[:, :-1]).T
#f = lambda theta: -obj_fn(theta, S_t, S1_t)


# > Processing system size 200 neurons
#   [Info] Estimating EP (nsamples: 360322, lr: 0.000500, patience: 30, % with transition: 0.998449)...
#   [Result took 22.283140] EP tst/full: 0.24623 0.48643 | R: 18.40379 | EP tst/R: 0.01338

if args.obs == 1:
    data = ep_estimators.RawDataset(S_t, S1_t)
elif args.obs == 2:
    data = ep_estimators.RawDataset2(S_t, S1_t)
else:
    raise ValueError(f"Invalid observable type {args.obs}. Use 1 or 2.")

lr=0.000500
trn, tst = data.split_train_test()
res=ep_estimators.get_EP_GradientAscent(data=trn, holdout_data=tst, lr=lr, verbose=2, report_every=10, patience=30 )
sigma = res.tst_objective
print("EP:", sigma)

# Extract coupling matrix θ
theta = res.theta.cpu().numpy()
if args.obs == 2:
    th = theta.reshape(N, N)
else:
    # Upper triangular matrix
    th = np.zeros((N, N), dtype=theta.dtype)
    triu_indices = np.triu_indices(N, k=1)
    th[triu_indices[0], triu_indices[1]] = theta
    th=th-th.T

# --- Area names ---
area_names, area_start_indices = np.unique(areas, return_index=True)
area_end_indices = np.r_[area_start_indices[1:], [len(areas)]]
area_centers = (area_start_indices + area_end_indices) / 2

# --- Cluster neurons based on θ ---
dist_matrix = np.abs(th) + np.abs(th.T)
np.fill_diagonal(dist_matrix, 0)
condensed_dist = squareform(dist_matrix)
linkage_matrix = linkage(condensed_dist, method='average')
sorted_indices = leaves_list(linkage_matrix)

th_sorted = th[sorted_indices, :][:, sorted_indices]

# --- Save and re-plot sorted matrix with improved visualization ---
areas = areas[sorted_indices]
inds3 = np.argsort(areas)
areas = areas[inds3]
th = th_sorted[inds3, :][:, inds3]

area_names, area_start_indices = np.unique(areas, return_index=True)
area_end_indices = np.r_[area_start_indices[1:], [len(areas)]]
area_centers = (area_start_indices + area_end_indices) / 2

# Define symmetric logarithmic normalization for better contrast,
# especially helpful when θ values vary widely around zero.

if args.fancy_color_scale:
    norm = mcolors.SymLogNorm(
        linthresh=0.012,    # Linear range around zero
        linscale=0.05,     # Controls the size of the linear region
        vmin=-np.max(np.abs(th)),  # Symmetric color scale limits
        vmax=np.max(np.abs(th))
    )
else:
    norm = None

# Create a new figure with specified size
plt.figure(figsize=(5, 4))

# Plot the θ matrix using red-white-blue colormap for signed values
# 'equal' ensures square pixels; 'nearest' avoids interpolation artifacts
plt.imshow(th, cmap='bwr', aspect='equal', interpolation='nearest', norm=norm)

# Add a label to the left side of the plot as a "title"
plt.text(-0.25, 0.5, r'$\theta_{ij}^*$', fontsize=18, 
         va='center', ha='center', rotation=0, transform=plt.gca().transAxes)

# Draw dotted lines to separate different brain areas visually
for idx in area_start_indices[1:]:  # Skip the first index (0)
    plt.axhline(idx - 0.5, color='k', linewidth=1, linestyle=':')
    plt.axvline(idx - 0.5, color='k', linewidth=1, linestyle=':')

# Compute the center positions of each area block for labeling
area_centers = (area_start_indices + area_end_indices) / 2

# Set axis tick positions and labels based on area centers
plt.xticks(area_centers, area_names, rotation=90)
plt.yticks(area_centers, area_names)

# Adjust tick label size for better readability
plt.tick_params(axis='both', which='major', labelsize=14)

# Add colorbar and adjust colorbar ticks
cbar = plt.colorbar(shrink=0.85)

tick_positions = [-1.0e-1,-0.5e-1,-0.25e-1, 0.0 ,0.25e-1,0.5e-1,1.e-1]
tick_labels = [r'$-0.1$', r'$-0.05$', r'$-0.025$', r'$0$', r'$0.025$', r'$0.05$', r'$0.1$']

cbar.set_ticks(tick_positions)              # Only these positions
cbar.set_ticklabels(tick_labels)            # Custom labels
cbar.ax.tick_params(length=6, width=1.5)     # Optional: adjust tick bar style
cbar.ax.minorticks_off()   


# Automatically adjust layout to minimize clipping
plt.tight_layout()
plt.savefig('img/Fig2b.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

