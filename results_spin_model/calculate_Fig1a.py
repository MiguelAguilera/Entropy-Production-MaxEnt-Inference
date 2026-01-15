import os
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt

# Import EP estimation routine
from get_spin_EP import *

# -------------------------------
# Main Entry Point
# -------------------------------
if __name__ == "__main__":

    # -------------------------------
    # Argument Parsing
    # -------------------------------
    def int_or_none(v):
        if v.lower() == "none":
            return None
        return int(v)
        
    parser = argparse.ArgumentParser(description="Estimate EP for the spin model with varying beta values.")

    parser.add_argument("--rep", type=int, default=1_000_000,
                        help="Number of repetitions for the simulation (default: 1,000,000)")
    parser.add_argument("--N", type=int, default=100,
                        help="System size (default: 100)")
    parser.add_argument("--BASE_DIR", type=str, default="~/MaxEntData",
                        help="Base directory to store simulation results (default: '~/MaxEntData')")
    parser.add_argument("--beta_min", type=float, default=0,
                        help="Minimum beta value (default: 0)")
    parser.add_argument("--beta_max", type=float, default=4,
                        help="Maximum beta value (default: 4)")
    parser.add_argument("--num_beta", type=int, default=26,
                        help="Number of beta values to simulate (default: 26)")
    parser.add_argument("--J0", type=float, default=0.0,
                        help="Mean interaction coupling (default: 0.0)")
    parser.add_argument("--DJ", type=float, default=1.0,
                        help="Variance of the quenched disorder (default: 1.0)")
    parser.add_argument('--no_plot', action='store_true', default=False,
                        help='Disable plotting if specified')
    parser.add_argument("--patterns", type=int, default=None,
                        help="Hopfield pattern density (default: None)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing output files (default: 6)")
    parser.add_argument("--num_neighbors", type=int_or_none, default=6,
                        help="Number of neighbors for sparse connectivity (default: None)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility (default: 0)")

    args = parser.parse_args()

    # -------------------------------
    # Global Setup
    # -------------------------------
    N = args.N
    rep = args.rep
    BASE_DIR = os.path.expanduser(args.BASE_DIR)
    betas = np.linspace(args.beta_min, args.beta_max, args.num_beta)

    # -------------------------------
    # Create Data Directory
    # -------------------------------
    SAVE_DATA_DIR = 'ep_data'
    if not os.path.exists(SAVE_DATA_DIR):
        print(f'Creating output directory: {SAVE_DATA_DIR}')
        os.makedirs(SAVE_DATA_DIR, exist_ok=True)

    # -------------------------------
    # Run Experiments Across Beta Values
    # -------------------------------
    EP = np.zeros((4, args.num_beta))  # Rows: [Empirical, MTUR, Newton-1, Gradient]

    for ib, beta in enumerate(np.round(betas, 8)):
        # Construct input and output file names
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

        # Perform EP estimation
        EP[:, ib] = calc(N, beta, rep, file_name, file_name_out, overwrite=args.overwrite, seed=args.seed)

    # -------------------------------
    # Plot Results
    # -------------------------------
    import seaborn as sns
    sns.set(style='white', font_scale=1.8)
    plt.rc('text', usetex=True)
    plt.rc('font', size=14, family='serif', serif=['latin modern roman'])
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm,newtxtext,newtxmath}')

    labels = [
        r'$\Sigma$',                   # Empirical EP
        r'$\Sigma_{\bm g}^\textnormal{\small TUR}$',  # MTUR
        r'$\widehat{\Sigma}_{\bm g}$',                # Newton-1
        r'${\Sigma}_{\bm g}$'                         # Gradient Descent
    ]

    cmap = plt.get_cmap('inferno_r')
    colors = [cmap(0.25), cmap(0.5), cmap(0.75)]

    fig, ax = plt.subplots(figsize=(4, 4), layout='constrained')

    # Reference critical point (if relevant)
    beta_c = 1.3485

    # Plot results
    plt.plot(betas[0], EP[0, 0], 'k', linestyle=(0, (2, 3)), label=labels[0], lw=3)
    for i in range(1, EP.shape[0]):
        plt.plot(betas, EP[i, :], label=labels[i], color=colors[i - 1], lw=2)
    plt.plot(betas, EP[0, :], 'k', linestyle=(0, (2, 3)), lw=3)  # Re-plot empirical for emphasis

    # Axes and labels
    plt.axis([betas[0], betas[-1], 0, np.max(EP) * 1.05])
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'EP', rotation=0, labelpad=20)

    # Add legend
    legend = plt.legend(
        ncol=1,
        columnspacing=0.25,
        handlelength=1.0,
        handletextpad=0.25,
        labelspacing=0.25,
        loc='best'
    )

    # -------------------------------
    # Save Plot
    # -------------------------------
    IMG_DIR = 'img'
    if not os.path.exists(IMG_DIR):
        print(f'Creating image directory: {IMG_DIR}')
        os.makedirs(IMG_DIR)

    plt.savefig(f'{IMG_DIR}/Fig1a.pdf', bbox_inches='tight', pad_inches=0.1)

    # Show plot (unless disabled)
    if not args.no_plot:
        plt.show()

