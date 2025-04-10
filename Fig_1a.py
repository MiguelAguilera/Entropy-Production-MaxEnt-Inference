import os
import argparse
import numpy as np
import torch
import h5py
from matplotlib import pyplot as plt
from methods_EP_multipartite import *

# -------------------------------
# Argument parsing for script execution
# -------------------------------
parser = argparse.ArgumentParser(description="Parse input arguments for the script.")
parser.add_argument("--num_steps", type=int, default=2**7,
                    help="Number of simulation steps (default: 128).")
parser.add_argument("--rep", type=int, default=1_000_000,
                    help="Number of repetitions for the simulation (default: 1000000).")
parser.add_argument("--size", type=int, default=100,
                    help="System size (default: 100).")
parser.add_argument("--BASE_DIR", type=str, default="~/MaxEntData",
                    help="Base directory to store simulation results (default: '~/MaxEntData').")
parser.add_argument("--beta_min", type=float, default=0,
                    help="Minimum beta value (default: 0).")
parser.add_argument("--beta_max", type=float, default=4,
                    help="Maximum beta value (default: 4).")
parser.add_argument("--num_beta", type=int, default=101,
                    help="Number of beta values to simulate (default: 101).")
parser.add_argument("--J0", type=float, default=1.0,
                    help="Mean interaction coupling (default: 1.0).")
parser.add_argument("--DJ", type=float, default=0.5,
                    help="Variance of the quenched disorder (default: 0.5).")
args = parser.parse_args()

# -------------------------------
# Global parameters
# -------------------------------
BASE_DIR = os.path.expanduser(args.BASE_DIR)
DTYPE = 'float32'

# Generate array of beta values
betas = np.linspace(args.beta_min, args.beta_max, args.num_beta)

# -------------------------------
# Main calculation function
# -------------------------------
def calc(N, rep):
    """
    Compute entropy production rate (EPR) estimates using various methods
    for a spin system of size N and given beta.
    """
    print(f"\n** DOING SYSTEM SIZE {N} with beta {beta:.2f} **", flush=True)

    file_name = f"{BASE_DIR}/sequential/run_reps_{rep}_steps_{args.num_steps}_{N:06d}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}.h5"
    print(file_name, flush=True)

    with h5py.File(file_name, 'r') as f:
        J = f['J'][:]
    J_t = torch.from_numpy(J)

    S_Exp = S_TUR = S_N1 = S_N2 = 0

    for i in range(N):
        with h5py.File(file_name, 'r') as f:
            S_i = f[f'S_{i}'][:].astype(DTYPE) * 2 - 1  # Convert to ±1 spin values
        S_t = torch.from_numpy(S_i)

        if S_i.shape[1] <= 1:
            continue

        # Estimate EP using MTUR
        sig_MTUR, theta_lin, Da = get_EP_MTUR(S_t, rep, i)

        # Estimate actual entropy production
        sigma_exp = exp_EP_spin_model(Da, J_t, i)

        # First Newton-based estimation
        sig_N1, theta_lin, Da = get_EP_Newton(S_t, rep, theta_lin, Da, i)

        # Second Newton-based estimation
        sig_N2, theta_lin2 = get_EP_Newton2(S_t, rep, theta_lin, Da, i)

        # Accumulate results
        S_Exp += sigma_exp
        S_TUR += sig_MTUR
        S_N1  += sig_N1
        S_N2  += sig_N2

    print(f"EPR (Experimental):  {S_Exp:.6f}")
    print(f"EPR (MTUR):          {S_TUR:.6f}")
    print(f"EPR (1-step Newton): {S_N1:.6f}")
    print(f"EPR (2-step Newton): {S_N2:.6f}")
    return np.array([S_Exp, S_TUR, S_N1, S_N2])

# -------------------------------
# Run experiments over beta values
# -------------------------------
EPR = np.zeros((4, args.num_beta))  # Rows: Exp, MTUR, Pert, Pert2

for ib, beta in enumerate(np.round(betas, 5)):
    EPR[:, ib] = calc(args.size, args.rep)
    filename = f'data/spin/maxent_beta_{beta}_r_{args.rep}_N_{args.size}.npz'
    np.savez(filename, EPR=EPR[:, ib])


# -------------------------------
# Plotting results
# -------------------------------
plt.rc('text', usetex=True)
font = {'size': 22, 'family': 'serif', 'serif': ['latin modern roman']}
plt.rc('font', **font)
plt.rc('legend', fontsize=20)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')

labels = [r'$\Sigma$', r'$\Sigma_{\bm g}^\textnormal{\small TUR}$', r'$\widehat{\Sigma}_{\bm g}$', r'${\Sigma}_{\bm g}$']  # Labels for comparison methods

cmap = plt.get_cmap('inferno_r')
colors = [cmap(0.25),cmap(0.5),cmap(0.75)]

plt.figure(figsize=(4, 4))

# Plot experimental EPR
plt.plot(betas, args.size * EPR[0, :], 'k', linestyle=(0, (1, 3)), label=labels[0], lw=2)

# Plot estimators
for i in range(1,4):
    plt.plot(betas, args.size * EPR[i, :], label=labels[i], color=colors[i], lw=2)

# Axis and labels
plt.axis([betas[0], betas[-1], 0, args.size * np.max(EPR) * 1.05])
plt.ylabel(r'$\Sigma$', rotation=0, labelpad=20)
plt.xlabel(r'$\beta$')

# Legend formatting
plt.legend(
    ncol=1,
    columnspacing=0.5,
    handlelength=1.0,
    handletextpad=0.5,
    loc='best'
)

# Save and show figure
plt.savefig('img/EP_spin_model.pdf', bbox_inches='tight')
plt.show()
