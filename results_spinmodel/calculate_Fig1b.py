import os, sys
import argparse
import numpy as np
import h5py
import hdf5plugin
from matplotlib import pyplot as plt
import seaborn as sns

sys.path.insert(0, '..')
from get_spin_EP import *

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
    file_name = f"{BASE_DIR}/sequential/run_reps_{rep}_steps_N_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}_num_neighbors_{args.num_neighbors}.npz"
    file_name_out = f"{SAVE_DATA_DIR}/results_N_{N}_reps_{rep}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}_num_neighbors_{args.num_neighbors}.h5"
else:
    file_name = f"{BASE_DIR}/sequential/run_reps_{rep}_N_{N:06d}_beta_{beta}_patterns_{args.patterns}.npz"
    file_name_out = f"{SAVE_DATA_DIR}/results_N_{N}_reps_{rep}_beta_{beta}_patterns_{args.patterns}.h5"
print(f"[Loading] Reading data from file:\n  → {file_name}\n")
        
EP, theta_N1,theta_N2, J =  calc(N, beta, rep, file_name, file_name_out, return_parameters=True, overwrite=args.overwrite)
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
>>>>>>> e07f36c8ea37b03f527d8689e63eeb30250fe63b
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


def R2(delta_theta, beta_delta_w):
    residual = delta_theta - beta_delta_w
    return 1 - np.mean(residual**2) / np.var(delta_theta)


if __name__ == "__main__":

    # -------------------------------
    # Argument Parsing
    # -------------------------------
    parser = argparse.ArgumentParser(description="Estimate EP and MaxEnt parameters for the spin model.")
    parser.add_argument("--rep", type=int, default=1_000_000)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--BASE_DIR", type=str, default="~/MaxEntData")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--J0", type=float, default=1.0)
    parser.add_argument("--DJ", type=float, default=0.5)
    parser.add_argument("--patterns", type=int, default=None)
    parser.add_argument("--num_neighbors", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    # -------------------------------
    # Matplotlib settings for LaTeX rendering
    # -------------------------------
    plt.rc('text', usetex=True)
    plt.rc('font', size=22, family='serif', serif=['latin modern roman'])
    plt.rc('legend', fontsize=20)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')

    # -------------------------------
    # Initialization and Parameters
    # -------------------------------
    BASE_DIR = os.path.expanduser(args.BASE_DIR)
    SAVE_DATA_DIR = 'ep_data/spin'
    N = args.N
    rep = args.rep
    beta = np.round(args.beta, 8)

    print(f'** DOING SYSTEM SIZE {N} with beta {beta:.6f} **', flush=True)

    if args.patterns is None:
        if args.num_neighbors is None:
            file_name = f"{BASE_DIR}/sequential/run_reps_{rep}_N_{N:06d}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}.npz"
            file_name_out = f"{SAVE_DATA_DIR}/results_N_{N}_reps_{rep}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}.h5"
        else:
            file_name = f"{BASE_DIR}/sequential/run_reps_{rep}_N_{N:06d}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}_num_neighbors_{args.num_neighbors}.npz"
            file_name_out = f"{SAVE_DATA_DIR}/results_N_{N}_reps_{rep}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}_num_neighbors_{args.num_neighbors}.h5"
    else:
        file_name = f"{BASE_DIR}/sequential/run_reps_{rep}_N_beta_{beta}_patterns_{args.patterns}.npz"
        file_name_out = f"{SAVE_DATA_DIR}/results_N_{N}_reps_{rep}_beta_{beta}_patterns_{args.patterns}.h5"

    print(f"[Loading] Reading data from file:\n  → {file_name}\n")

    # -------------------------------
    # Load and Compute
    # -------------------------------
    EP, theta_N1, theta_N2, J = calc(N, beta, rep, file_name, file_name_out,
                                     return_parameters=True, overwrite=args.overwrite)
    dbetaJ = beta * (J - J.T)

    theta_N1 = expand_offdiag(theta_N1)
    theta_N2 = expand_offdiag(theta_N2)

    dtheta_N1 = theta_N1 - theta_N1.T
    dtheta_N2 = theta_N2 - theta_N2.T

    upper_indices = np.triu_indices(N, k=1)
    nonzero_mask = dbetaJ[upper_indices] != 0
    filtered_indices = (upper_indices[0][nonzero_mask], upper_indices[1][nonzero_mask])

    r2_N1 = R2(dtheta_N1[filtered_indices], dbetaJ[filtered_indices])
    r2_N2 = R2(dtheta_N2[filtered_indices], dbetaJ[filtered_indices])
    print(f"R² (dbetaJ vs. dtheta_Gaussian): {r2_N1:.4f}")
    print(f"R² (dbetaJ vs. dtheta_Newton): {r2_N2:.4f}")

    # -------------------------------
    # Visualization
    # -------------------------------
    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = plt.get_cmap('inferno_r')

    sns.scatterplot(
        x=dbetaJ[filtered_indices],
        y=dtheta_N2[filtered_indices],
        color=cmap(0.75), s=10, alpha=0.7, rasterized=True
    )

    dbetaJ_min, dbetaJ_max = np.min(dtheta_N2), np.max(dtheta_N2)
    plt.plot([dbetaJ_min, dbetaJ_max], [dbetaJ_min, dbetaJ_max], 'k', linestyle='dashed')
    plt.axis([dbetaJ_min, dbetaJ_max, dbetaJ_min, dbetaJ_max])

    ticks = np.arange(-4, 5, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    plt.xlabel(r"$\beta(w_{ij} - w_{ji})$")
    plt.ylabel(r'$\theta_{ij}-\theta_{ji}$', rotation=90, labelpad=0)
    plt.savefig('img/Fig_1b.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()

