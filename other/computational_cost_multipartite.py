import time, os, gc, random, sys
import numpy as np
from matplotlib import pyplot as plt
import torch
import h5py

sys.path.insert(0, '..')

import spin_model
import observables
import ep_estimators
import utils

from tqdm import tqdm


# ============================================================
# Config
# ============================================================
utils.set_default_torch_device()

DATA_DIR = 'data'
IMG_DIR  = 'img'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

H5_PATH = os.path.join(DATA_DIR, 'results.h5')
OVERWRITE = False  # set True to force recompute for matching entries

# Sweep over system size
Ns = [10, 15, 20, 25, 30, 35, 40]

# Sweep over samples per spin
Ss = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

Ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

k = 6            # avg number of neighbors in sparse coupling matrix
beta = 2.0       # inverse temperature
N_for_Ss = 40    # N used in the S-sweep

R = 100            # <-- number of repeats per case
BASE_SEED = 1234   # base seed; per-repeat seeds are derived as BASE_SEED + r


# ============================================================
# Helpers
# ============================================================
def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def _trial_path(N: int, samples_per_spin: int, beta: float, k: int, seed: int) -> str:
    """
    Unique HDF5 group path for a single trial, keyed by all inputs + seed (so repeats don't collide).
    """
    return f"/trials/N={int(N)}/S={int(samples_per_spin)}/beta={float(beta)}/k={int(k)}/seed={int(seed)}"

def _trial_complete(g: h5py.Group) -> bool:
    """Check that the trial group has all required scalar datasets."""
    required = ("sigma_emp", "sigma_mp", "sigma_nmp", "t_mp", "t_nmp")
    for k_ in required:
        if k_ not in g:
            return False
        if getattr(g[k_], "shape", None) != ():
            return False
    return True

def get_or_run(N: int, samples_per_spin: int, beta: float, k: int, seed: int, overwrite: bool=False):
    """
    If a matching trial exists in results.h5 and overwrite=False -> load it.
    Else run the experiment, write it, and return the values.
    """
    path = _trial_path(N, samples_per_spin, beta, k, seed)
    with h5py.File(H5_PATH, "a") as h5:
        if (not overwrite) and path in h5:
            g = h5[path]
            if isinstance(g, h5py.Group) and _trial_complete(g):
                # Load
                sigma_emp  = float(g["sigma_emp"][()])
                sigma_mp   = float(g["sigma_mp"][()])
                sigma_nmp  = float(g["sigma_nmp"][()])
                t_mp       = float(g["t_mp"][()])
                t_nmp      = float(g["t_nmp"][()])
                print(f"[load] {path}")
                return sigma_emp, sigma_mp, sigma_nmp, t_mp, t_nmp
            else:
                print(f"[recompute:incomplete] {path}")

        # Compute
        print(f"[compute] {path}")
        vals = run_single_experiment(N=N, samples_per_spin=samples_per_spin, beta=beta, k=k)

        # Write scalars (replace any stale children)
        g = h5.require_group(path)
        for name in list(g.keys()):
            del g[name]
        sigma_emp, sigma_mp, sigma_nmp, t_mp, t_nmp = vals
        g.create_dataset("sigma_emp", data=np.array(sigma_emp))
        g.create_dataset("sigma_mp",  data=np.array(sigma_mp))
        g.create_dataset("sigma_nmp", data=np.array(sigma_nmp))
        g.create_dataset("t_mp",      data=np.array(t_mp))
        g.create_dataset("t_nmp",     data=np.array(t_nmp))
        # annotate
        g.attrs["N"] = int(N)
        g.attrs["samples_per_spin"] = int(samples_per_spin)
        g.attrs["beta"] = float(beta)
        g.attrs["k"] = int(k)
        g.attrs["seed"] = int(seed)
        g.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

        return vals


def run_single_experiment(N: int, samples_per_spin: int, beta: float, k: int):
    """
    Runs one full experiment (simulate + EP estimates) and returns:
    (sigma_emp, sigma_mp, sigma_nmp, t_mp, t_nmp)
    """
    stime = time.time()

    # Couplings and simulation
    J = spin_model.get_couplings_random(N=N, k=k)  # keep sparse-k version
    S, F = spin_model.run_simulation(beta=beta, J=J, samples_per_spin=samples_per_spin)
    num_samples_per_spin, N_eff = S.shape
    assert N_eff == N
    total_flips = N * num_samples_per_spin
    print(f"Sampled {total_flips} transitions in {time.time()-stime:.3f}s")

    # Empirical EP
    stime = time.time()
    sigma_emp = spin_model.get_empirical_EP(beta, J, S, F)
    time_emp = time.time() - stime

    # Multipartite EP (sum over spins with weights p_i)
    sigma_mp = 0.0
    t_mp = 0.0
    for i in range(N):
        p_i = F[:, i].sum() / total_flips
        g_samples = observables.get_g_observables(S, F, i)
        data = observables.Dataset(g_samples=g_samples)
        train, val, test = data.split_train_val_test()

        stime = time.time()
        spin_g, _ = ep_estimators.get_EP_Estimate(data, validation=val, test=test)
        sigma_mp += p_i * spin_g
        t_mp += time.time() - stime

    # Cleanup GPU between estimators
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    # Non-multipartite EP (cross-correlations)
    X0, X1 = spin_model.convert_to_nonmultipartite(S, F)
    X0 = X0.astype(np.float32)
    X1 = X1.astype(np.float32)
    dataS = observables.CrossCorrelations2(X0, X1)

    train, val, test = dataS.split_train_val_test()

    stime = time.time()
    sigma_nmp, _ = ep_estimators.get_EP_Estimate(train, validation=val, test=test)
    t_nmp = time.time() - stime

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    print(f"\nEntropy production estimates (N={N}, k={k}, β={beta})")
    print(f"  Σ     (Empirical)                                : {sigma_emp:.6f}  ({time_emp:.3f}s)")
    print(f"  Σ_g   (Multipartite optimization, gradient asc.) : {sigma_mp:.6f}  ({t_mp:.3f}s)")
    print(f"  Σ_g   (Regular optimization, gradient asc.)      : {sigma_nmp:.6f}  ({t_nmp:.3f}s)")
    print()

    return sigma_emp, sigma_mp, sigma_nmp, t_mp, t_nmp

def summarize_over_repeats(results_RxM, error_mode="std"):
    """
    results_RxM: np.array shape (R, M)
    Returns mean (M,), error (M,) where error is std or sem.
    """
    mean = results_RxM.mean(axis=0)
    if results_RxM.shape[0] > 1:
        std = results_RxM.std(axis=0, ddof=1)
        if error_mode == "std":
            err = std
        elif error_mode == "sem":
            err = std / np.sqrt(results_RxM.shape[0])
        else:
            raise ValueError(f"Unknown error_mode={error_mode}, must be 'std' or 'sem'")
    else:
        err = np.zeros(results_RxM.shape[1])
    return mean, err

def plot_with_fill(x, y_mean, y_err, label, use_fill):
    plt.plot(x, y_mean, 'o-', label=label)
    if use_fill and np.any(y_err > 0):
        plt.fill_between(x, y_mean - y_err, y_mean + y_err, alpha=0.2)

# -------------------------------
# Plot styling helpers
# -------------------------------
def setup_matplotlib_style():
    import seaborn as sns
    sns.set(style='white', font_scale=1.8)
    plt.rc('text', usetex=True)
    plt.rc('font', size=14, family='serif', serif=['latin modern roman'])
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm,newtxtext}')


def style_axes(ax, xlabel, ylabel, ypad=20):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, labelpad=ypad)
    ax.legend(
        ncol=1,
        columnspacing=0.25,
        handlelength=1.0,
        handletextpad=0.25,
        labelspacing=0.25,
        loc='best'
    )

def plot_series(ax, x, y_mean, y_err, label, color=None, lw=2, marker=None, fill=True):
    line, = ax.plot(x, y_mean, label=label, lw=lw, marker=marker, color=color)
    if fill and (y_err is not None) and np.any(y_err > 0):
        ax.fill_between(x, y_mean - y_err, y_mean + y_err, alpha=0.2, color=line.get_color())
    return line



# ============================================================
# Sweep 1: vary N (fixed samples_per_spin)
# ============================================================
samples_per_spin_fixed = 20000

M = len(Ns)
emp_RxM   = np.zeros((R, M))
mp_RxM    = np.zeros((R, M))
nmp_RxM   = np.zeros((R, M))
tmp_RxM   = np.zeros((R, M))  # time mp
tnmp_RxM  = np.zeros((R, M))  # time nmp

for idx in tqdm(range(R * len(Ns)), desc="Sweep over N"):
    r = idx // len(Ns)
    j = idx % len(Ns)
    N = Ns[j]
    seed = BASE_SEED + r*1000 + j
    seed_everything(seed)

    sigma_emp, sigma_mp, sigma_nmp, t_mp, t_nmp = get_or_run(
        N=N, samples_per_spin=samples_per_spin_fixed, beta=beta, k=k, seed=seed, overwrite=OVERWRITE
    )
    emp_RxM[r, j]  = sigma_emp
    mp_RxM[r, j]   = sigma_mp
    nmp_RxM[r, j]  = sigma_nmp
    tmp_RxM[r, j]  = t_mp
    tnmp_RxM[r, j] = t_nmp

# choose error measure globally
ERROR_MODE = "std"   # "sem" or "std"

emp_mean, emp_err   = summarize_over_repeats(emp_RxM, error_mode=ERROR_MODE)
mp_mean, mp_err     = summarize_over_repeats(mp_RxM,  error_mode=ERROR_MODE)
nmp_mean, nmp_err   = summarize_over_repeats(nmp_RxM, error_mode=ERROR_MODE)
tmp_mean, tmp_err   = summarize_over_repeats(tmp_RxM,  error_mode=ERROR_MODE)
tnmp_mean, tnmp_err = summarize_over_repeats(tnmp_RxM, error_mode=ERROR_MODE)

# ============================================================
# Styled plots + saving (after computing means/stds)
# ============================================================
setup_matplotlib_style()
cmap = plt.get_cmap('inferno_r')
colors = [cmap(0.5), cmap(0.75)]  # two estimator curves

# 1) EP vs N
fig, ax = plt.subplots(figsize=(4, 4))
# Empirical (thick dashed black)
ax.plot(Ns, emp_mean, 'k', linestyle=(0, (2, 3)), lw=3, label=r'$\Sigma$')
# Estimators
plot_series(ax, Ns, mp_mean,  mp_err,  r'multipartite', color=colors[1], lw=2, fill=(R > 1))
plot_series(ax, Ns, nmp_mean, nmp_err, r'regular',      color=colors[0], lw=2, fill=(R > 1))
# Re-plot empirical on top for emphasis
ax.plot(Ns, emp_mean, 'k', linestyle=(0, (2, 3)), lw=3)
ax.set_xlim([Ns[0], Ns[-1]])
ax.set_ylim([0, 1.05 * max(np.max(emp_mean), np.max(mp_mean), np.max(nmp_mean))])
style_axes(ax, r'Number of spins', 'EP')
plt.savefig(os.path.join(IMG_DIR, 'Fig_N_EP.pdf'), bbox_inches='tight', pad_inches=0.1)

# 2) Time vs N
fig, ax = plt.subplots(figsize=(4, 4))
# No empirical time; plot the two estimators
plot_series(ax, Ns, tmp_mean,  tmp_err,  r'multipartite', color=colors[1], lw=2, fill=(R > 1))
plot_series(ax, Ns, tnmp_mean, tnmp_err, r'regular',      color=colors[0], lw=2, fill=(R > 1))
ax.set_xlim([Ns[0], Ns[-1]])
ax.set_ylim([0, 1.05 * max(np.max(tmp_mean), np.max(tnmp_mean))])
style_axes(ax, r'Number of spins', 'Time (s)', ypad=12)
plt.savefig(os.path.join(IMG_DIR, 'Fig_N_Time.pdf'), bbox_inches='tight', pad_inches=0.1)



# ============================================================
# Sweep 2: vary samples_per_spin (fixed N)
# ============================================================
M = len(Ss)
emp_RxM   = np.zeros((R, M))
mp_RxM    = np.zeros((R, M))
nmp_RxM   = np.zeros((R, M))
tmp_RxM   = np.zeros((R, M))
tnmp_RxM  = np.zeros((R, M))

for idx in tqdm(range(R * len(Ss)), desc="Sweep over S"):
    r = idx // len(Ss)
    j = idx % len(Ss)
    Nsamples = Ss[j]
    seed = BASE_SEED + 10_000 + r*1000 + j
    seed_everything(seed)

    sigma_emp, sigma_mp, sigma_nmp, t_mp, t_nmp = get_or_run(
        N=N_for_Ss, samples_per_spin=Nsamples, beta=beta, k=k, seed=seed, overwrite=OVERWRITE
    )
    emp_RxM[r, j]  = sigma_emp
    mp_RxM[r, j]   = sigma_mp
    nmp_RxM[r, j]  = sigma_nmp
    tmp_RxM[r, j]  = t_mp
    tnmp_RxM[r, j] = t_nmp

emp_mean, emp_err   = summarize_over_repeats(emp_RxM)
mp_mean, mp_err     = summarize_over_repeats(mp_RxM)
nmp_mean, nmp_err   = summarize_over_repeats(nmp_RxM)
tmp_mean, tmp_err   = summarize_over_repeats(tmp_RxM)
tnmp_mean, tnmp_err = summarize_over_repeats(tnmp_RxM)



# ============================================================
# Styled plots + saving (after computing means/stds)
# ============================================================
setup_matplotlib_style()
cmap = plt.get_cmap('inferno_r')
colors = [cmap(0.5), cmap(0.75)]  # two estimator curves

# 3) EP vs S
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(Ss, emp_mean, 'k', linestyle=(0, (2, 3)), lw=3, label=r'$\Sigma$')
plot_series(ax, Ss, mp_mean,  mp_err,  r'multipartite', color=colors[1], lw=2, fill=(R > 1))
plot_series(ax, Ss, nmp_mean, nmp_err, r'regular',      color=colors[0], lw=2, fill=(R > 1))
ax.plot(Ss, emp_mean, 'k', linestyle=(0, (2, 3)), lw=3)
ax.set_xlim([Ss[0], Ss[-1]])
ax.set_ylim([0, 1.05 * max(np.max(emp_mean), np.max(mp_mean), np.max(nmp_mean))])
style_axes(ax, r'Number of samples', 'EP')
plt.savefig(os.path.join(IMG_DIR, 'Fig_S_EP.pdf'), bbox_inches='tight', pad_inches=0.1)

# 4) Time vs S
fig, ax = plt.subplots(figsize=(4, 4))
plot_series(ax, Ss, tmp_mean,  tmp_err,  r'multipartite', color=colors[1], lw=2, fill=(R > 1))
plot_series(ax, Ss, tnmp_mean, tnmp_err, r'regular',      color=colors[0], lw=2, fill=(R > 1))
ax.set_xlim([Ss[0], Ss[-1]])
ax.set_ylim([0, 1.05 * max(np.max(tmp_mean), np.max(tnmp_mean))])
style_axes(ax, r'Number of samples', 'Time (s)', ypad=12)
plt.savefig(os.path.join(IMG_DIR, 'Fig_S_Time.pdf'), bbox_inches='tight', pad_inches=0.1)

plt.show()

