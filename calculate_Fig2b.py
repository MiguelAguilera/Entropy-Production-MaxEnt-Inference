import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import matplotlib.colors as mcolors
import h5py
import hdf5plugin
from methods_EP_parallel import *

parser = argparse.ArgumentParser(description="Estimate EP and MaxEnt parameters for a neuropixels dataset.")
parser.add_argument("--BASE_DIR", type=str, default="~/Neuropixels",
                    help="Base directory to store the data (default: '~/Neuropixels').")
args = parser.parse_args()


# --- Constants and Configuration ---
BASE_DIR = Path(args.BASE_DIR).expanduser()
DTYPE = 'float32'
tol_per_param = 1e-6
bin_size = 0.01
session_id = 8
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
S_t = torch.from_numpy(S[:, 1:])
S1_t = torch.from_numpy(S[:, :-1])
f = lambda theta: -obj_fn(theta, S_t, S1_t)

args = get_torchmin_args(S_t, tol_per_param)
args['x0'] = torch.zeros((N * (N - 1)) // 2)  # Upper-triangular vector
args['lambda_'] = 0
res = minimize2(f, **args)

# Extract coupling matrix θ
theta = res.x.numpy()
th = np.zeros((N, N), dtype=theta.dtype)
triu_indices = np.triu_indices(N, k=1)
th[triu_indices[0], triu_indices[1]] = theta
th=th-th.T
sigma = -res.fun
print("Log-likelihood:", sigma)

# --- Area names ---
area_names, area_start_indices = np.unique(areas, return_index=True)
area_end_indices = np.r_[area_start_indices[1:], [len(areas)]]
area_centers = (area_start_indices + area_end_indices) / 2

# --- Cluster neurons based on θ ---
th_abs = np.abs(th)

th_sign = np.sign(th)
th_custom = np.where(th_sign == th_sign.T, -th_abs, 0)  # Optional: not used directly

condensed_dist = squareform(th_abs)
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
norm = mcolors.SymLogNorm(
    linthresh=0.008,    # Linear range around zero
    linscale=0.05,     # Controls the size of the linear region
    vmin=-np.max(np.abs(th)),  # Symmetric color scale limits
    vmax=np.max(np.abs(th))
)

# Create a new figure with specified size
plt.figure(figsize=(5, 4))

# Plot the θ matrix using red-white-blue colormap for signed values
# 'equal' ensures square pixels; 'nearest' avoids interpolation artifacts
plt.imshow(th, cmap='bwr', aspect='equal', interpolation='nearest', norm=norm)

# Add a label to the left side of the plot as a "title"
plt.text(-0.35, 0.5, r'$\theta_{ij}$', fontsize=22, 
         va='center', ha='center', transform=plt.gca().transAxes)

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
tick_positions = [-1e-1,-0.5e-1,-0.25e-1, 0,0.25e-1,0.5e-1, 1e-1]
tick_labels = [r'$-0.1$', r'$-0.25$', r'$-0.5$', r'$0$', r'$0.25$', r'$0.5$', r'$0.1$']
cbar.set_ticks(tick_positions)              # Only these positions
cbar.set_ticklabels(tick_labels)            # Custom labels
cbar.ax.tick_params(length=6, width=1.5)     # Optional: adjust tick bar style
cbar.ax.minorticks_off()   


# Automatically adjust layout to minimize clipping
plt.tight_layout()
plt.savefig('img/Fig2b.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

