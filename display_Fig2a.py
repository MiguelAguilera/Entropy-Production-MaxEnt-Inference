import numpy as np
import matplotlib.pyplot as plt
import argparse

# ========================
# Argument Parser
# ========================

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
                    help="Normalize EP by firing rate (default: True).")

parser.add_argument("--remove_outliers", action="store_true", default=False,
                    help="Remove outliers from EP values (default: False).")

parser.add_argument("--types", nargs="+", default=["active", "passive", "gabor"],
                    help="Session types to include.")

args = parser.parse_args()


# ========================
# Plotting Configuration
# ========================

plt.rc('text', usetex=True)
font = {'size': 14, 'family': 'serif', 'serif': ['latin modern roman']}
plt.rc('font', **font)
plt.rc('legend', fontsize=12)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')


# ========================
# Main Execution
# ========================

if __name__ == "__main__":

    # Use parsed arguments
    mode = args.mode
    L2 = args.L2
    order = args.order
    bin_size = args.bin_size
    sizes = args.sizes
    normalize = args.normalize
    remove_outliers_flag = args.remove_outliers
    types = args.types
    rep = args.R

    num_sessions = 103

    # Data containers
    EP = {session_type: {size: [] for size in sizes} for session_type in types}
    _loaded_sessions = {}

    def load_EP(size, session_type, session_id, r):
        """
        Loads EP for the given settings and caches it.
        """
        global _loaded_sessions
        key = (session_type, session_id, r)

        if key not in _loaded_sessions:
            filename = f'data/neuropixels/neuropixels_{mode}_{order}_{session_type}_id_{session_id}_binsize_{bin_size}_L2_{L2}_rep_{r}.npz'
            try:
                data = np.load(filename)
                _loaded_sessions[key] = {
                    'EP': data['EP'],
                    'R': data['R'],
                    'sizes': data['sizes']
                }
            except Exception:
                _loaded_sessions[key] = None

        session_data = _loaded_sessions[key]
        if session_data is None:
            return None

        try:
            index = list(session_data['sizes']).index(size)
            EP = session_data['EP'][index]
            R = session_data['R'][index]
            return EP / R if normalize else EP
        except:
            return None

    def remove_outliers(data):
        """
        IQR-based outlier removal.
        """
        if len(data) < 3:
            return data
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return [x for x in data if lower_bound <= x <= upper_bound]

    # Load EP data
    for r in range(rep):
        for session_id in range(num_sessions):
            for size in sizes:
                for session_type in types:
                    EP_value = load_EP(size, session_type, session_id, r)
                    if EP_value is not None and not np.isnan(EP_value):
                        EP[session_type][size].append(EP_value)

    # Outlier filtering
    if remove_outliers_flag:
        EP = {
            session_type: {
                size: remove_outliers(values)
                for size, values in EP[session_type].items()
            }
            for session_type in types
        }

    # Scatter plot
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('inferno_r')
    colors = list(cmap(np.linspace(0, 1, len(types) + 2)[1:-1][::-1]))

    for session_type, color in zip(types, colors):
        for size in sizes:
            plt.scatter(
                [size] * len(EP[session_type][size]),
                EP[session_type][size],
                color=color, alpha=0.6,
                label=f'{session_type.capitalize()} N={size}'
            )

    plt.xlabel('System Size (N)')
    plt.ylabel('EP Value')

    # Compute stats
    mean_EP = {
        session_type: [0] + [np.mean(EP[session_type][size]) for size in sizes]
        for session_type in types
    }

    std_EP = {
        session_type: [0] + [np.std(EP[session_type][size]) for size in sizes]
        for session_type in types
    }

    sem_EP = {
        session_type: [0] + [
            np.std(EP[session_type][size]) / np.sqrt(len(EP[session_type][size]))
            if len(EP[session_type][size]) > 1 else 0
            for size in sizes
        ]
        for session_type in types
    }

    # Line plot with SEM
    plt.figure(figsize=(5, 2))
    markers = ['-s', '-o', '-^']

    for session_type, color, marker in zip(types, colors, markers):
        x_values = [0] + sizes
        y_values = mean_EP[session_type]
        y_err = sem_EP[session_type]

        plt.errorbar(
            x_values, y_values, yerr=y_err,
            color=color, label=f'{session_type.capitalize()}',
            fmt=marker, capsize=5
        )

        plt.fill_between(
            x_values,
            np.array(y_values) - np.array(y_err),
            np.array(y_values) + np.array(y_err),
            color=color, alpha=0.3
        )

    upper_y = max([max(mean_EP[t]) + max(std_EP[t]) for t in types])
    plt.axis([0, max(sizes), 0, 1.2 * upper_y])

    plt.xlabel(r'$N$')
    ylabel = r'$\dfrac{\Sigma_{\bm g}}{R}$' if normalize else r'$\dfrac{\Sigma_{\bm g}}{N}$'
    plt.ylabel(ylabel, rotation=0, labelpad=20)

    plt.legend(
        ncol=1, framealpha=1,
        columnspacing=0.5,
        handlelength=1., handletextpad=0.5,
        bbox_to_anchor=(0.03, 0.55)
    )

    plt.savefig('img/Fig2a.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()

