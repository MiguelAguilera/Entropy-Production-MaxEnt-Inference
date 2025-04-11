import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Plot EP results from file.")

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

args = parser.parse_args()

# -------------------------------
# Load results
# -------------------------------
filename = f"data/spin/data_Fig_1a_rep_{args.rep}_steps_{args.num_steps}_N_{args.size}_J0_{args.J0}_DJ_{args.DJ}_betaMin_{args.beta_min}_betaMax_{args.beta_max}_numBeta_{args.num_beta}.npz"
data = np.load(filename)
EP = data["EP"]
betas = data["betas"]

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
    r'${\Sigma}_{\bm g}$'
]

cmap = plt.get_cmap('inferno_r')
colors = [cmap(0.25), cmap(0.5), cmap(0.75)]

plt.figure(figsize=(4, 4))

plt.plot(betas[0], EP[0, 0], 'k', linestyle=(0, (2, 3)), label=labels[0], lw=3)  # Reference line
for i in range(1, 4):
    plt.plot(betas, EP[i, :], label=labels[i], color=colors[i-1], lw=2)
plt.plot(betas, EP[0, :], 'k', linestyle=(0, (2, 3)), lw=3)  # Re-plot experimental

plt.axis([betas[0], betas[-1], 0, np.max(EP) * 1.05])
plt.ylabel(r'$\Sigma$', rotation=0, labelpad=20)
plt.xlabel(r'$\beta$')

plt.legend(
    ncol=1,
    columnspacing=0.5,
    handlelength=1.0,
    handletextpad=0.5,
    loc='best'
)

plt.savefig('img/Fig_1a.pdf', bbox_inches='tight')
plt.show()

