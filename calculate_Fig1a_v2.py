import os, argparse, time
import numpy as np
import torch
import h5py
import hdf5plugin # This needs to be imported even thought its not explicitly used
from matplotlib import pyplot as plt
from methods_EP_multipartite import *
import gd 

GD_MODE = 2  # 0 for no GD
             # 1 for pytorch Adam
             # 2 for our Adam


# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Estimate EP for the spin model with varying beta values.")

parser.add_argument("--num_steps", type=int, default=2**7,
                    help="Number of simulation steps (default: 128)")
parser.add_argument("--rep", type=int, default=1_000_000,
                    help="Number of repetitions for the simulation (default: 1,000,000)")
parser.add_argument("--size", type=int, default=100,
                    help="System size (default: 100)")
parser.add_argument("--BASE_DIR", type=str, default="~/MaxEntData",
                    help="Base directory to store simulation results (default: '~/MaxEntData').")
parser.add_argument("--beta_min", type=float, default=0,
                    help="Minimum beta value (default: 0)")
parser.add_argument("--beta_max", type=float, default=4,
                    help="Maximum beta value (default: 4)")
parser.add_argument("--num_beta", type=int, default=101,
                    help="Number of beta values to simulate (default: 101)")
parser.add_argument("--J0", type=float, default=1.0,
                    help="Mean interaction coupling (default: 1.0)")
parser.add_argument("--DJ", type=float, default=0.5,
                    help="Variance of the quenched disorder (default: 0.5)")
parser.add_argument('--no_plot', action='store_true', default=False,
                    help='Disable plotting if specified')
parser.add_argument("--patterns", type=int, default=None,
                    help="Hopfield pattern density (default: None).")
args = parser.parse_args()

N = args.size
rep = args.rep

# -------------------------------
# Global Setup
# -------------------------------
BASE_DIR = os.path.expanduser(args.BASE_DIR)
DTYPE = 'float32'
betas = np.linspace(args.beta_min, args.beta_max, args.num_beta)

# -------------------------------
# Main Entropy Production Calculation Function
# -------------------------------
def calc(N, rep):
    """
    Compute entropy production rate (EP) estimates using multiple methods for a spin system.

    Parameters:
        N (int): System size.
        rep (int): Number of repetitions.

    Returns:
        np.ndarray: EP estimates [empirical, MTUR, Newton-1, Newton-2]
    """
    print()
    print("=" * 70)
    print(f"  Starting EP estimation | System size: {N} | β = {beta:.4f}")
    print("=" * 70)

    if args.patterns is None:
        file_name = f"{BASE_DIR}/sequential/run_reps_{rep}_steps_{args.num_steps}_{N:06d}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}.npz"
    else:
        file_name = f"{BASE_DIR}/sequential/run_reps_{rep}_steps_{args.num_steps}_{N:06d}_beta_{beta}_patterns_{args.patterns}.npz"
    print(f"[Loading] Reading data from file:\n  → {file_name}\n")

    data = np.load(file_name)
    J = data['J']
    H = data['H']
    assert(np.all(H==0))  # We do not support local fields in our analysis

    # with h5py.File(file_name, 'r') as f:
    #     J = f['J'][:]
    #     H = f['H'][:]
    #     assert(np.all(H==0))  # We do not support local fields in our analysis

    J_t = torch.from_numpy(J)

    # Initialize accumulators
    S_Emp = S_TUR = S_N1 = S_GD= 0
    # S_N2 = 0
    
    T = N * rep  # Total spin-flip attempts

    time_gd  = 0
    start_time = time.time()

    for i in range(N):
        #with h5py.File(file_name+'.h5', 'r') as f:
        #    S_i = f[f'S_{i}'][:].astype(DTYPE) * 2 - 1  # Convert to ±1
        F_i = data["F"][i]
        S_i = data["S"][:, F_i].astype("float32") * 2 - 1  # convert {0,1} → {-1,1}

        S_i_t = torch.from_numpy(S_i)

        if S_i.shape[1] <= 10:
            print(f"  [Warning] Skipping spin {i}: insufficient time steps")
            continue

        Pi=S_i.shape[1]/T
        # Estimate entropy production using various methods
        sig_N1, theta1, Da          = get_EP_Newton(S_i_t, i)
        sig_MTUR                    = get_EP_MTUR(S_i_t, i)
        sigma_emp                   = exp_EP_spin_model(Da, J_t[i,:], i)
        # sig_N2, theta2              = get_EP_Newton2(S_i_t, theta1, Da, i)

        if GD_MODE > 0:
            start_time_gd_i = time.time()
            if GD_MODE == 2:
                sig_GD, theta_gd        = get_EP_Adam(S_i_t, theta_init=theta1, Da=Da, i=i) 
            elif GD_MODE == 1:
                sig_GD, theta_gd = gd.get_EP_gd(S_i_t, i, x0=theta1,  num_iters=1000)
            else:
                raise Exception('Uknown GD_MODE')
            time_gd += time.time() - start_time_gd_i
                
        else:
            sig_GD = np.nan

        # Aggregate results
        S_Emp += Pi*sigma_emp
        S_TUR += Pi*sig_MTUR
        S_N1  += Pi*sig_N1
        # S_N2  += Pi*max(sig_N1,sig_N2)
        S_GD  += Pi*sig_GD


    print(f"\n[Results] {time.time()-start_time:3f}s")
    print(f"  EP (Empirical)    :    {S_Emp:.6f}")
    print(f"  EP (MTUR)         :    {S_TUR:.6f}")
    print(f"  EP (1-step Newton):    {S_N1:.6f}")
    # print(f"  EP (2-step Newton):    {S_N2:.6f}")
    print(f"  EP (Grad Ascent  ):    {S_GD:.6f}   {time_gd:3f}s")
    print("-" * 70)

    return np.array([S_Emp, S_TUR, S_N1, S_N2, S_GD])

# -------------------------------
# Run Experiments Across Beta Values
# -------------------------------
EP = np.zeros((4, args.num_beta))  # Rows: Empirical, MTUR, Newton-1, GradientAscent

for ib, beta in enumerate(np.round(betas, 8)):
    EP[:, ib] = calc(N, rep)
    
# -------------------------------
# Save results
# -------------------------------
SAVE_DATA_DIR = 'ep_data/spin'
if not os.path.exists(SAVE_DATA_DIR):
    print(f'Creating base directory: {SAVE_DATA_DIR}')
    os.makedirs(SAVE_DATA_DIR)
filename = f"{SAVE_DATA_DIR}/data_Fig_1a_rep_{rep}_steps_{args.num_steps}_N_{N}_J0_{args.J0}_DJ_{args.DJ}_betaMin_{args.beta_min}_betaMax_{args.beta_max}_numBeta_{args.num_beta}.npz"
np.savez(filename, EP=EP, betas=betas)
print(f'Saved calculations to {filename}')

if not args.no_plot:
    # -------------------------------
    # Plot Results
    # -------------------------------
    plt.rc('text', usetex=True)
    plt.rc('font', size=22, family='serif', serif=['latin modern roman'])
    plt.rc('legend', fontsize=20)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')

    labels = [
        r'$\Sigma$', 
        r'$\Sigma_{\bm g}^\textnormal{\small TUR}$', 
        r'$\widehat{\Sigma}_{\bm g}$', 
        # r'${\Sigma}_{\bm g}$',
        r'${\Sigma}_{\bm g}$',
    ]

    cmap = plt.get_cmap('inferno_r')
    colors = [cmap(0.25), cmap(0.5), cmap(0.75),cmap(0.95)]

    plt.figure(figsize=(4, 4))

    # Plot each EP estimator
    plt.plot(betas[0], EP[0, 0], 'k', linestyle=(0, (2, 3)), label=labels[0], lw=3)  # Reference line
    for i in range(1, EP.shape[0]):
        plt.plot(betas, EP[i, :], label=labels[i], color=colors[i-1], lw=2)
    plt.plot(betas, EP[0, :], 'k', linestyle=(0, (2, 3)), lw=3)  # Re-plot empirical for clarity

    # Axes and labels
    plt.axis([betas[0], betas[-1], 0, np.nanmax(EP) * 1.05])
    plt.ylabel(r'$\Sigma$', rotation=0, labelpad=20)
    plt.xlabel(r'$\beta$')

    # Legend
    plt.legend(
        ncol=1,
        columnspacing=0.5,
        handlelength=1.0,
        handletextpad=0.5,
        loc='best'
    )

    # Save and show figure
    plt.savefig('img/Fig_1a.pdf', bbox_inches='tight')
    plt.show()

