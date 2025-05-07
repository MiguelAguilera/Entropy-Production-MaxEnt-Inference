import argparse, sys, os
import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import matplotlib.colors as mcolors


parser = argparse.ArgumentParser(description='Load coupling coefficients from a NumPy file')
parser.add_argument('filename', type=str, help='Path to the .npz file containing the coupling coefficients')
parser.add_argument("--fancy_color_scale", dest="fancy_color_scale", action="store_true",
                    help="Fancy color rescaling.")
args = parser.parse_args()

data = np.load(args.filename)

# Extract the arrays
areas = data['areas']
th = data['th']
sigma = data['sigma']


# Configure LaTeX rendering in matplotlib
plt.rc('text', usetex=True)
plt.rc('font', size=18, family='serif', serif=['latin modern roman'])
plt.rc('legend', fontsize=18)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')


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

vmax = np.percentile(np.abs(th), 99.5)  # Use 95th percentile instead of max
if args.fancy_color_scale:
    norm = mcolors.SymLogNorm(
        linthresh=0.012,    # Linear range around zero
        linscale=0.05,     # Controls the size of the linear region
        vmin=-vmax,  # Symmetric color scale limits
        vmax=vmax
    )
else:
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

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

