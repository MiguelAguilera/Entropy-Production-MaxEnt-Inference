import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import h5py
import hdf5plugin  # noqa: F401

# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Aggregate and plot EP results from Neuropixels data.")

parser.add_argument("--mode", type=str, default="visual", choices=["visual", "nonvisual", "all"])
parser.add_argument("--L2", type=str, default="0")
parser.add_argument("--order", type=str, default="random", choices=["random", "sorted"])
parser.add_argument("--bin_size", type=float, default=0.01)
parser.add_argument("--R", type=int, default=10)
parser.add_argument("--sizes", nargs="+", type=int, default=[50, 100, 150, 200, 250, 300])
parser.add_argument("--no_normalize", action="store_true", default=False)
parser.add_argument("--remove_outliers", action="store_true", default=False)
parser.add_argument("--types", nargs="+", default=["active", "passive", "gabor"])
parser.add_argument("--obs", type=int, default=1)
parser.add_argument("--lr", type=float, default=1)
parser.add_argument("--lr_scale", type=str, choices=["none", "N", "sqrtN"], default="N")

args = parser.parse_args()

# -------------------------------
# Plot Configuration
# -------------------------------
import seaborn as sns
sns.set(style='white', font_scale=1.4)
plt.rc('text', usetex=True)
plt.rc('font', size=12, family='serif', serif=['latin modern roman'])
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm,newtxtext,newtxmath}')

# -------------------------------
# Helper Functions
# -------------------------------
def load_EP(size, session_type, session_id, r, normalize=True):
    key = (session_type, session_id, r)
    filename = f'ep_data/neuropixels_{args.mode}_{args.order}_binsize_{args.bin_size}_obs_{args.obs}_BB_lr_{args.lr}_lr-scale_{args.lr_scale}.h5'

    try:
        with h5py.File(filename, 'r') as f:
            group_path = f"{session_type}/{session_id}/rep_{r}"
            if group_path not in f:
                return None
            grp = f[group_path]
            sizes = grp['sizes'][:]
            if size not in sizes:
                return None
            idx = list(sizes).index(size)
            ep = grp['EP'][idx]
            rate = grp['R'][idx]
            return ep / rate if normalize else ep
    except Exception as e:
        print(f"[Warning] Failed to load data for {key}: {e}")
        return None

def remove_outliers(data):
    if len(data) < 3:
        return data
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [x for x in data if lower <= x <= upper]

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    normalize = not args.no_normalize
    sizes = args.sizes
    types = args.types
    rep = args.R
    num_sessions = 103

    EP = {stype: {size: [] for size in sizes} for stype in types}

    # Aggregate EP values
    for r in range(rep):
        for session_id in range(num_sessions):
            for stype in types:
                for size in sizes:
                    val = load_EP(size, stype, session_id, r, normalize=normalize)
                    if val is not None and not np.isnan(val):
                        EP[stype][size].append(val)

    if args.remove_outliers:
        for stype in types:
            for size in sizes:
                EP[stype][size] = remove_outliers(EP[stype][size])

    # Scatter plot
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('viridis_r')
    colors = list(cmap(np.linspace(0, 1, len(types) + 2)[1:-1][::-1]))

    for stype, color in zip(types, colors):
        for size in sizes:
            plt.scatter([size] * len(EP[stype][size]), EP[stype][size], color=color, alpha=0.6, label=f'{stype.capitalize()} N={size}')

    plt.xlabel('System Size (N)')
    plt.ylabel('EP Value')

    # Summary statistics
    mean_EP = {stype: [0] + [np.mean(EP[stype][s]) for s in sizes] for stype in types}
    sem_EP = {stype: [0] + [np.std(EP[stype][s]) / np.sqrt(len(EP[stype][s])) if len(EP[stype][s]) > 1 else 0 for s in sizes] for stype in types}

    # Line plot with SEM
    plt.figure(figsize=(5, 2))
    markers = ['-s', '-o', '-^']

    for stype, color, marker in zip(types, colors, markers):
        x_vals = [0] + sizes
        y_vals = mean_EP[stype]
        y_err = sem_EP[stype]
        plt.errorbar(x_vals, y_vals, yerr=y_err, color=color, label=stype.capitalize(), fmt=marker, capsize=5)
        plt.fill_between(x_vals, np.array(y_vals) - np.array(y_err), np.array(y_vals) + np.array(y_err), color=color, alpha=0.3)

    max_size = max(sizes)
    ymax = max([max(mean_EP[t]) for t in types]) * 1.1
    plt.axis([0, max_size, 0, ymax])
    plt.xlabel(r'$N$')
    plt.ylabel('EP / spike' if normalize else 'EP')
    plt.legend(loc='lower right')

    IMG_DIR = 'img'
    os.makedirs(IMG_DIR, exist_ok=True)
    plt.savefig(f'{IMG_DIR}/Fig2a.pdf', bbox_inches='tight')
    plt.show()
