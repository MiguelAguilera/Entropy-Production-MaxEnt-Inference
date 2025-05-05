import os
import argparse
import numpy as np
import torch
import h5py
import hdf5plugin
from matplotlib import pyplot as plt
import seaborn as sns
from get_spin_EP import *

# -------------------------------
# Argument parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Estimate EP and MaxEnt parameters for the spin model.")
parser.add_argument("--num_steps", type=int, default=2**7,
                    help="Number of simulation steps (default: 128).")
parser.add_argument("--rep", type=int, default=1_000_000,
                    help="Number of repetitions for the simulation (default: 1000000).")
parser.add_argument("--size", type=int, default=100,
                    help="System size (default: 100).")
parser.add_argument("--BASE_DIR", type=str, default="~/MaxEntData",
                    help="Base directory to store simulation results (default: '~/MaxEntData').")
parser.add_argument("--beta", type=float, default=1.3484999614126383 ,
                    help="beta value (default: 1.3484999614126383 ).")
parser.add_argument("--J0", type=float, default=1.0,
                    help="Mean interaction coupling (default: 1.0).")
parser.add_argument("--DJ", type=float, default=0.5,
                    help="Variance of the quenched disorder (default: 0.5).")
parser.add_argument("--patterns", type=int, default=None,
                    help="Hopfield pattern density (default: None).")
parser.add_argument("--num_neighbors", type=int, default=None,
                    help="Number of neighbors for sparse connectivity (default: None).")
parser.add_argument("--overwrite", action="store_true",  default=False,
                    help="Do not overwrite existing files.")
args = parser.parse_args()

# -------------------------------
# Matplotlib settings for LaTeX rendering
# -------------------------------
plt.rc('text', usetex=True)
plt.rc('font', size=22, family='serif', serif=['latin modern roman'])
plt.rc('legend', fontsize=20)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')

# -------------------------------
# Initialization and parameters
# -------------------------------
BASE_DIR = os.path.expanduser(args.BASE_DIR)
DTYPE = 'float32'
N = args.size
rep = args.rep

beta = np.round(args.beta, 8) # Inverse temperature (interaction strength)

print(f'** DOING SYSTEM SIZE {N} with beta {beta:.6f} **', flush=True)

SAVE_DATA_DIR = 'ep_data/spin'

# -------------------------------
# Load data
# -------------------------------

if args.patterns is None:
    file_name = f"{BASE_DIR}/sequential/run_reps_{rep}_steps_{args.num_steps}_{N:06d}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}_num_neighbors_{args.num_neighbors}.npz"
    file_name_out = f"{SAVE_DATA_DIR}/results_N_{N}_reps_{rep}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}_num_neighbors_{args.num_neighbors}.h5"
else:
    file_name = f"{BASE_DIR}/sequential/run_reps_{rep}_steps_{args.num_steps}_{N:06d}_beta_{beta}_patterns_{args.patterns}.npz"
    file_name_out = f"{SAVE_DATA_DIR}/results_N_{N}_reps_{rep}_beta_{beta}_patterns_{args.patterns}.h5"
print(f"[Loading] Reading data from file:\n  → {file_name}\n")
        
EP, theta_N1,theta_N2, J =  calc(N, rep, file_name, file_name_out, return_parameters=True, overwrite=args.overwrite)
print(theta_N1.shape)
dJ = J-J.T

# -------------------------------
# Save results
# -------------------------------
#filename=f'data/spin/data_Fig_1b.npz'
#np.savez(filename, theta_N1=theta_N1.copy(), theta_N2=theta_N2.copy(), dJ=dJ)

# -------------------------------
# Helper function to expand off-diagonal matrix
# -------------------------------
def expand_offdiag(th):
    """
    Expand an Nx(N-1) off-diagonal matrix into a full NxN matrix with zeros on the diagonal.
    """
    N = th.shape[0]
    full = np.zeros((N, N))
    for i in range(N):
        full[i, :i] = th[i, :i]
        full[i, i+1:] = th[i, i:]
    return full

# -------------------------------
# Symmetrize and compute error metrics
# -------------------------------
theta_N1 = expand_offdiag(theta_N1)
theta_N2 = expand_offdiag(theta_N2)

dtheta_N1 = theta_N1 - theta_N1.T
dtheta_N2 = theta_N2 - theta_N2.T


upper_indices = np.triu_indices(N, k=1)  # Upper triangle indices


def R2(delta_theta, beta_delta_w):
    residual = delta_theta - beta_delta_w
    return 1-np.mean(residual**2) / np.var(delta_theta)
r2_N1 = R2(dtheta_N1[upper_indices],J[upper_indices])
r2_N2 = R2(dtheta_N2[upper_indices],J[upper_indices])
print(f"R² (dJ vs. dtheta_Gaussian): {r2_N1:.4f}")
print(f"R² (dJ vs. dtheta_Newton): {r2_N2:.4f}")

# -------------------------------
# Visualization
# -------------------------------

plt.figure(figsize=(4, 4))

cmap = plt.get_cmap('inferno_r')
colors = [cmap(0.25),cmap(0.5),cmap(0.75)]

# Scatter plot with Seaborn aesthetics
sns.scatterplot(x=dJ[upper_indices], y=dtheta_N1[upper_indices], color=cmap(0.5), s=10, alpha=0.7,rasterized=True)
sns.scatterplot(x=dJ[upper_indices], y=dtheta_N2[upper_indices], color=cmap(0.75), s=10, alpha=0.7,rasterized=True)
sns.scatterplot(x=np.ones(2)*100, y=np.ones(2)*100, label=r'$\bm{\hat\theta}$', color=cmap(0.5), s=20)
sns.scatterplot(x=np.ones(2)*100, y=np.ones(2)*100,label=r'$\bm \theta^*$', color=cmap(0.75), s=20)

# Add reference line
dJ_min, dJ_max = np.min(dJ), np.max(dJ)
plt.plot([dJ_min, dJ_max], [dJ_min, dJ_max], 'k', linestyle='dashed')
plt.axis([dJ_min*1.05, dJ_max*1.05,dJ_min*1.05, dJ_max*1.05])
# Labels and title
plt.xlabel(r"$\beta(w_{ij} - w_{ji})$")
plt.ylabel(r'$\theta_{ij}-\theta_{ji}$', rotation=90, labelpad=-5)
#plt.title(r"Comparison of $J_{ij} - J_{ji}$ vs. $\theta_{ij}$", fontsize=18)
#plt.legend()
plt.legend(
    ncol=1,
    columnspacing=0.5,   # Reduce space between columns
    handlelength=1.,    # Shorten the length of legend lines
    handletextpad=0.5,   # Reduce space between line and label
    loc='best'           # You can adjust the location if needed,
)
plt.savefig('img/Fig_1b.pdf', bbox_inches='tight', pad_inches=0)

plt.show()

