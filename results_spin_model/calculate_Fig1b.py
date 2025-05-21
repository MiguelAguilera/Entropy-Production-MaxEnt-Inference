import os
import sys
import argparse
import numpy as np
import h5py
import hdf5plugin
from matplotlib import pyplot as plt
import seaborn as sns

# Import EP estimators from parent directory
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
# Helper Functions
# -------------------------------

def expand_offdiag(th):
    """
    Expands an Nx(N-1) matrix to a full NxN matrix with zeros on the diagonal.

    Args:
        th (np.ndarray): Nx(N-1) matrix (e.g., theta values excluding diagonals).

    Returns:
        np.ndarray: Full NxN matrix with zeros on the diagonal.
    """
    N = th.shape[0]
    full = np.zeros((N, N))
    for i in range(N):
        full[i, :i] = th[i, :i]
        full[i, i+1:] = th[i, i:]
    return full


def R2(delta_theta, beta_delta_w):
    """
    Compute R² (coefficient of determination) between estimated and true antisymmetric couplings.

    Args:
        delta_theta (np.ndarray): Estimated antisymmetric part of theta.
        beta_delta_w (np.ndarray): Ground truth antisymmetric part of beta-weighted J matrix.

    Returns:
        float: R² score.
    """
    residual = delta_theta - beta_delta_w
    return 1 - np.mean(residual**2) / np.var(delta_theta)


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    # -------------------------------
    # Argument Parsing
    # -------------------------------
    parser = argparse.ArgumentParser(description="Estimate EP and MaxEnt parameters for the spin model.")

    parser.add_argument("--rep", type=int, default=1_000_000)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--BASE_DIR", type=str, default="~/MaxEntData")
    parser.add_argument("--beta", type=float, default=2.5)
    parser.add_argument("--J0", type=float, default=0.0)
    parser.add_argument("--DJ", type=float, default=1.0)
    parser.add_argument("--patterns", type=int, default=None)
    parser.add_argument("--num_neighbors", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--no_plot", action="store_true", default=False,
                        help="Disable plotting")

    args = parser.parse_args()

    # -------------------------------
    # Setup
    # -------------------------------
    BASE_DIR = os.path.expanduser(args.BASE_DIR)
    SAVE_DATA_DIR = 'ep_data'
    if not os.path.exists(SAVE_DATA_DIR):
        print(f'Creating base directory: {SAVE_DATA_DIR}')
        os.makedirs(SAVE_DATA_DIR, exist_ok=True)

    N = args.N
    rep = args.rep
    beta = np.round(args.beta, 8)

    print(f'** Running system with size N={N}, β={beta:.6f} **', flush=True)

    # -------------------------------
    # Construct File Paths
    # -------------------------------
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
    # Load data and run estimation
    # -------------------------------
    EP, theta_N1, theta_N2, J = calc(
        N, beta, rep, file_name, file_name_out,
        return_parameters=True, overwrite=args.overwrite, seed=args.seed
    )

    # Compute antisymmetric matrix beta * (J - J.T)
    dbetaJ = beta * (J - J.T)

    # Expand off-diagonal estimates
    theta_N1 = expand_offdiag(theta_N1)
    theta_N2 = expand_offdiag(theta_N2)

    # Compute antisymmetric parts of theta
    dtheta_N1 = theta_N1 - theta_N1.T
    dtheta_N2 = theta_N2 - theta_N2.T

    # Use only upper triangular (i < j) values to avoid redundancy
    upper_indices = np.triu_indices(N, k=1)
    nonzero_mask = dbetaJ[upper_indices] < 1e10
    filtered_indices = (upper_indices[0][nonzero_mask], upper_indices[1][nonzero_mask])

    # -------------------------------
    # Quality Scores
    # -------------------------------
    r2_N1 = R2(dtheta_N1[filtered_indices], dbetaJ[filtered_indices])
    r2_N2 = R2(dtheta_N2[filtered_indices], dbetaJ[filtered_indices])
    print(f"R² (dbetaJ vs. dtheta_Gaussian): {r2_N1:.4f}")
    print(f"R² (dbetaJ vs. dtheta_Newton)  : {r2_N2:.4f}")

    # -------------------------------
    # Plotting
    # -------------------------------
    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = plt.get_cmap('inferno_r')

    sns.scatterplot(
        x=dbetaJ[filtered_indices],
        y=dtheta_N2[filtered_indices],
        color=cmap(0.75), s=10, alpha=0.7, rasterized=True
    )

    # Identity line for reference
    dbetaJ_min, dbetaJ_max = np.min(dtheta_N2), np.max(dtheta_N2)
    plt.plot([dbetaJ_min, dbetaJ_max], [dbetaJ_min, dbetaJ_max], 'k', linestyle='dashed')
    plt.axis([dbetaJ_min, dbetaJ_max, dbetaJ_min, dbetaJ_max])

    # Set ticks and labels
    ticks = np.arange(-4, 5, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(r"$\beta(w_{ij} - w_{ji})$")
    ax.set_ylabel(r'$\theta_{ij}-\theta_{ji}$', rotation=90, labelpad=0)

    # Save figure
    IMG_DIR = 'img'
    if not os.path.exists(IMG_DIR):
        print(f'Creating image directory: {IMG_DIR}')
        os.makedirs(IMG_DIR)

    plt.savefig(f'{IMG_DIR}/Fig1b.pdf')

    # Show if not disabled
    if not args.no_plot:
        plt.show()

