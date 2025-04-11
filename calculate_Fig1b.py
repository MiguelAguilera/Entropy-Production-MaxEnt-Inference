import os
import argparse
import numpy as np
import torch
import h5py
import hdf5plugin
from matplotlib import pyplot as plt
import seaborn as sns
from methods_EP_multipartite import *  # Assumes methods.py contains exp_EP_spin_model, get_MTUR, get_Pert, get_Pert2

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
parser.add_argument("--critical_beta", type=float, default=1.3484999614126383 ,
                    help="Critical beta value (default: 1.3484999614126383 ).")
parser.add_argument("--num_beta", type=int, default=101,
                    help="Number of beta values to simulate (default: 101).")
parser.add_argument("--J0", type=float, default=1.0,
                    help="Mean interaction coupling (default: 1.0).")
parser.add_argument("--DJ", type=float, default=0.5,
                    help="Variance of the quenched disorder (default: 0.5).")
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

beta = np.round(args.critical_beta, 8) # Inverse temperature (interaction strength)

print(f'** DOING SYSTEM SIZE {N} with beta {beta:.6f} **', flush=True)



# -------------------------------
# Load data
# -------------------------------

file_name = f"{BASE_DIR}/sequential/run_reps_{rep}_steps_{args.num_steps}_{N:06d}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}.h5"
print(file_name, flush=True)

with h5py.File(file_name, 'r') as f:
    J = f['J'][:]
J_t = torch.from_numpy(J)

# -------------------------------
# Extract antisymmetric part of interaction matrix
# -------------------------------

mask = ~torch.eye(N, dtype=torch.bool)  # Mask for off-diagonal elements
dJ = ((J - J.T)[mask]).reshape(N, N - 1)

# -------------------------------
# Compute entropy production and surrogate models
# -------------------------------

S_Exp = S_TUR = S_N1 = S_N2 = 0
th1 = np.zeros((N,N-1))
th2 = np.zeros((N,N-1))
for i in range(N):
    with h5py.File(file_name, 'r') as f:
        S_i = f[f'S_{i}'][:].astype(DTYPE) * 2 - 1  # Convert to ±1 spin values
    S_t = torch.from_numpy(S_i)

    if S_i.shape[1] <= 1:
        continue

    # Estimate EP using different methods
    sig_N1, sig_MTUR, theta1, Da = get_EP_Newton(S_t, rep, i)
    sigma_exp = exp_EP_spin_model(Da, J_t, i)
    sig_N2, theta2 = get_EP_Newton2(S_t, rep, theta1, Da, i)

    # Accumulate results
    S_Exp += sigma_exp
    S_TUR += sig_MTUR
    S_N1  += sig_N1
    S_N2  += sig_N2
    
    th1[i,:] = theta1.numpy() 
    th2[i,:] = theta1.numpy() + theta2.numpy()

# -------------------------------
# Save results
# -------------------------------
filename=f'data/spin/data_Fig_1b.npz'
np.savez(filename, th1=th1.copy(), th2=th2.copy(), dJ=dJ)

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
th1 = expand_offdiag(th1)
th2 = expand_offdiag(th2)
dJ = expand_offdiag(dJ)

dth1 = th1 - th1.T
dth2 = th2 - th2.T

# -------------------------------
# Visualization
# -------------------------------

plt.figure(figsize=(4, 4))

upper_indices = np.triu_indices(N, k=1)  # Upper triangle indices

cmap = plt.get_cmap('inferno_r')
colors = [cmap(0.25),cmap(0.5),cmap(0.75)]

# Scatter plot with Seaborn aesthetics
sns.scatterplot(x=dJ[upper_indices], y=dth1[upper_indices], color=cmap(0.5), s=10, alpha=0.7,rasterized=True)
sns.scatterplot(x=dJ[upper_indices], y=dth2[upper_indices], color=cmap(0.75), s=10, alpha=0.7,rasterized=True)
sns.scatterplot(x=np.ones(2)*100, y=np.ones(2)*100, label=r'$\hat\Sigma_{\bm g}$', color=cmap(0.5), s=20)
sns.scatterplot(x=np.ones(2)*100, y=np.ones(2)*100,label=r'$\Sigma_{\bm g}$', color=cmap(0.75), s=20)

# Add reference line
dJ_min, dJ_max = np.min(dJ), np.max(dJ)
plt.plot([dJ_min, dJ_max], [dJ_min, dJ_max], 'k', linestyle='dashed')
plt.axis([dJ_min*1.05, dJ_max*1.05,dJ_min*1.05, dJ_max*1.05])
# Labels and title
plt.xlabel(r"$\beta(w_{ij} - w_{ji})$")
plt.ylabel(r'$\theta_{ij}^*-\theta_{ji}^*$', rotation=90, labelpad=-5)
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

