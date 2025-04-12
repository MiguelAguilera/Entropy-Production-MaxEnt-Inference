import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Aggregate and plot EP results from Neuropixels data.")

parser.add_argument("--mode", type=str, default="visual",
                    choices=["visual", "nonvisual", "all"],
                    help="Brain area mode to filter neurons (default: visual).")

parser.add_argument("--L2", type=str, default="0",
                    help="L2 regularization term used in the analysis (default: 0).")

parser.add_argument("--order", type=str, default="random",
                    choices=["random", "sorted"],
                    help="Ordering of neurons (default: random).")

parser.add_argument("--bin_size", type=float, default=0.01,
                    help="Bin size for spike binning (default: 0.01).")

parser.add_argument("--R", type=int, default=10,
                    help="Number of repetitions per session and size (default: 10).")

parser.add_argument("--sizes", nargs="+", type=int,
                    default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                    help="List of population sizes to analyze.")

parser.add_argument("--normalize", action="store_true", default=True, 
                    help="Normalize EP by firing rate.")

parser.add_argument("--remove_outliers", action="store_false", default=False,
                    help="Remove outliers from EP values.")

parser.add_argument("--types", nargs="+", default=["active", "passive", "gabor"],
                    help="Session types to include.")

args = parser.parse_args()

# --- Plot config ---
plt.rc('text', usetex=True)
plt.rc('font', size=14, family='serif', serif=['Latin Modern Roman'])
plt.rc('legend', fontsize=12)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')

# --- Configuration ---
DTYPE = 'float32'
sizes = args.sizes
types = args.types
num_sessions = 102

# Initialize container
EP = {session_type: {size: [] for size in sizes} for session_type in types}

# --- Load EP data ---
def load_ep(size, session_type, session_id, r):
    filename = f'data/neuropixels/neuropixels_{args.mode}_{args.order}_{session_type}_id_{session_id}_binsize_{args.bin_size}_L2_{args.L2}_rep_{r}.npz'
    try:
        data = np.load(filename)
        ep = data['EP']
        rep = data['rep']
        sizes_arr = data['sizes']
        index = list(sizes_arr).index(size)
        print(args.normalize)
        if args.normalize:
            return ep[index] / R[index] if R[index] > 0 else None
        else:
            return ep[index]
    except Exception as e:
        return None

# --- Outlier removal (IQR) ---
def remove_outliers_iqr(data):
    if len(data) < 3:
        return data
    q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q3 - q1
    return [x for x in data if q1 - 1.5 * iqr <= x <= q3 + 1.5 * iqr]

# --- Aggregate data ---
for session_id in range(num_sessions):
    for session_type in types:
        for size in sizes:
            for r in range(args.rep):
                ep_value = load_ep(size, session_type, session_id, r)
                if ep_value is not None and not np.isnan(ep_value):
                    EP[session_type][size].append(ep_value)

# --- Remove outliers ---
if args.remove_outliers:
    EP = {
        session_type: {
            size: remove_outliers_iqr(values)
            for size, values in EP[session_type].items()
        }
        for session_type in types
    }

# --- Prepare plotting ---
cmap = plt.get_cmap('inferno_r')
colors = list(cmap(np.linspace(0, 1, len(types) + 2)[1:-1][::-1]))
markers = ['-s', '-o', '-^']

mean_EP = {
    session_type: [0] + [np.mean(EP[session_type][size]) for size in sizes]
    for session_type in types
}
sem_EP = {
    session_type: [0] + [
        np.std(EP[session_type][size], ddof=1) / np.sqrt(len(EP[session_type][size]))
        if len(EP[session_type][size]) > 1 else 0
        for size in sizes
    ]
    for session_type in types
}

# --- Plot ---
plt.figure(figsize=(5, 2))
for session_type, color, marker in zip(types, colors, markers):
    x_vals = [0] + sizes
    y_vals = mean_EP[session_type]
    y_errs = sem_EP[session_type]
    
    plt.errorbar(x_vals, y_vals, yerr=y_errs, color=color, label=f'{session_type.capitalize()}',
                 fmt=marker, capsize=5)
    plt.fill_between(x_vals, np.array(y_vals) - np.array(y_errs),
                     np.array(y_vals) + np.array(y_errs),
                     color=color, alpha=0.3)

# --- Axis and labels ---
max_y = max(max(mean_EP[st]) + max(sem_EP[st]) for st in types)
plt.axis([0, max(sizes), 0, 1.2 * max_y])
plt.xlabel(r'$N$')
ylabel = r'$\dfrac{\Sigma_{\bm g}}{R}$' if args.normalize else r'$\dfrac{\Sigma_{\bm g}}{N}$'
plt.ylabel(ylabel, rotation=0, labelpad=20)
plt.legend(ncol=1, framealpha=1, columnspacing=0.5, handlelength=1., handletextpad=0.5)

# --- Save and show ---
Path("img").mkdir(exist_ok=True)
output_name = f'img/EP_neuropixels_{args.mode}_L2_{args.L2}_{"norm" if args.normalize else "raw"}.pdf'
plt.savefig(output_name, bbox_inches='tight', pad_inches=0)
plt.show()

