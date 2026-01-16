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
Ss = [10000, 20000, 30000, 40000, 50000, 8000, 16000, 32000, 64000, 128000]
Ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

S_mp = [8_000, 16_000, 32_000, 64_000, 100_000, 200_000, 400_000, 600_000, 800_000, 1000_000]
N_mp =  [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

S_nmp = [20000, 40000, 60000, 80000, 100_000, 120_000]
N_nmp = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

k = 6            # avg number of neighbors in sparse coupling matrix
beta = 2.0       # inverse temperature
N_for_Ss = 40    # N used in the S-sweep

R = 1000            # <-- number of repeats per case
BASE_SEED = 1234   # base seed; per-repeat seeds are derived as BASE_SEED + r

# Modes
MP_MODE  = "mp"   # multipartite estimator
NMP_MODE = "nmp"  # non-multipartite (regular) estimator


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

def _trial_path(N: int, samples_per_spin: int, beta: float, k: int, seed: int, mode: str) -> str:
    """
    Unique HDF5 group path for a single trial and mode, so mp/nmp cache separately.
    """
    mode = str(mode)
    return (
        f"/trials/{mode}"
        f"/N={int(N)}/S={int(samples_per_spin)}/beta={float(beta)}/k={int(k)}/seed={int(seed)}"
    )

def _required_keys_for_mode(mode: str):
    if mode == MP_MODE:
        return ("sigma_emp", "sigma_mp", "t_mp")
    elif mode == NMP_MODE:
        return ("sigma_emp", "sigma_nmp", "t_nmp")
    else:
        raise ValueError(f"Unknown mode={mode!r}")

def _trial_complete(g: h5py.Group, mode: str) -> bool:
    """Check that the trial group has all required scalar datasets for the given mode."""
    required = _required_keys_for_mode(mode)
    for k_ in required:
        if k_ not in g:
            return False
        if getattr(g[k_], "shape", None) != ():
            return False
    return True

def get_or_run(N: int, samples_per_spin: int, beta: float, k: int, seed: int,
               mode: str, overwrite: bool=False):
    """
    Load (sigma_emp, sigma_est, t_est) for the given mode from cache, or compute and store.
    """
    path = _trial_path(N, samples_per_spin, beta, k, seed, mode)
    with h5py.File(H5_PATH, "a") as h5:
        if (not overwrite) and path in h5:
            g = h5[path]
            if isinstance(g, h5py.Group) and _trial_complete(g, mode):
                # Load
                sigma_emp = float(g["sigma_emp"][()])
                if mode == MP_MODE:
                    sigma_est = float(g["sigma_mp"][()])
                    t_est     = float(g["t_mp"][()])
                else:
                    sigma_est = float(g["sigma_nmp"][()])
                    t_est     = float(g["t_nmp"][()])
                print(f"[load] {path}")
                return sigma_emp, sigma_est, t_est
            else:
                print(f"[recompute:incomplete] {path}")

        # Compute
        print(f"[compute] {path}")
        sigma_emp, sigma_est, t_est = run_single_experiment(
            N=N, samples_per_spin=samples_per_spin, beta=beta, k=k, mode=mode
        )

        # Write scalars (replace any stale children)
        g = h5.require_group(path)
        for name in list(g.keys()):
            del g[name]
        g.create_dataset("sigma_emp", data=np.array(sigma_emp))
        if mode == MP_MODE:
            g.create_dataset("sigma_mp",  data=np.array(sigma_est))
            g.create_dataset("t_mp",      data=np.array(t_est))
        else:
            g.create_dataset("sigma_nmp", data=np.array(sigma_est))
            g.create_dataset("t_nmp",     data=np.array(t_est))
        # annotate
        g.attrs["mode"] = str(mode)
        g.attrs["N"] = int(N)
        g.attrs["samples_per_spin"] = int(samples_per_spin)
        g.attrs["beta"] = float(beta)
        g.attrs["k"] = int(k)
        g.attrs["seed"] = int(seed)
        g.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

        return sigma_emp, sigma_est, t_est


def run_single_experiment(N: int, samples_per_spin: int, beta: float, k: int, mode: str):
    """
    Runs one experiment (simulate + ONE EP estimate per `mode`) and returns:
    (sigma_emp, sigma_est, t_est)
    where:
      - if mode == 'mp'  : sigma_est = sigma_mp,  t_est = t_mp
      - if mode == 'nmp' : sigma_est = sigma_nmp, t_est = t_nmp
    """
    stime = time.time()

    # Couplings and simulation
    J = spin_model.get_couplings_random(N=N, k=k)  # keep sparse-k version
    S, F = spin_model.run_simulation(beta=beta, J=J, samples_per_spin=samples_per_spin)
    num_samples_per_spin, N_eff = S.shape
    assert N_eff == N
    total_flips = N * num_samples_per_spin
#    print(f"Sampled {total_flips} transitions in {time.time()-stime:.3f}s")

    # Empirical EP (always compute)
    stime = time.time()
    sigma_emp = spin_model.get_empirical_EP(beta, J, S, F)
    time_emp = time.time() - stime

    # Choose estimator branch
    if mode == MP_MODE:
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

        # Cleanup GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

#        print(f"\nEntropy production estimates (N={N}, k={k}, β={beta}) [mode=mp]")
#        print(f"  Σ     (Empirical)                                : {sigma_emp:.6f}  ({time_emp:.3f}s)")
#        print(f"  Σ_g   (Multipartite optimization, gradient asc.) : {sigma_mp:.6f}  ({t_mp:.3f}s)\n")
        return sigma_emp, sigma_mp, t_mp

    elif mode == NMP_MODE:
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

#        print(f"\nEntropy production estimates (N={N}, k={k}, β={beta}) [mode=nmp]")
#        print(f"  Σ     (Empirical)                                : {sigma_emp:.6f}  ({time_emp:.3f}s)")
#        print(f"  Σ_g   (Regular optimization, gradient asc.)      : {sigma_nmp:.6f}  ({t_nmp:.3f}s)\n")
        return sigma_emp, sigma_nmp, t_nmp

    else:
        raise ValueError(f"Unknown mode={mode!r}")


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
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm,newtxtext,newtxmath}')

def style_axes(ax, xlabel, ylabel, ypad=20, legend=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, labelpad=ypad)
    if legend:
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
# Utility: run a sweep for a given mode and x-grid
# ============================================================
def sweep_over_N(N_grid, samples_per_spin_fixed, mode: str, error_mode="std"):
    M = len(N_grid)
    emp_RxM  = np.zeros((R, M))
    est_RxM  = np.zeros((R, M))
    time_RxM = np.zeros((R, M))

    for idx in tqdm(range(R * M), desc=f"Sweep over N [{mode}]"):
        r = idx // M
        j = idx % M
        N = N_grid[j]
        seed = BASE_SEED + r*1000 + j
        seed_everything(seed)

        sigma_emp, sigma_est, t_est = get_or_run(
            N=N, samples_per_spin=samples_per_spin_fixed, beta=beta, k=k,
            seed=seed, mode=mode, overwrite=OVERWRITE
        )
        emp_RxM[r, j]  = sigma_emp
        est_RxM[r, j]  = sigma_est
        time_RxM[r, j] = t_est

    emp_mean, emp_err = summarize_over_repeats(emp_RxM,  error_mode=error_mode)
    est_mean, est_err = summarize_over_repeats(est_RxM,  error_mode=error_mode)
    t_mean,   t_err   = summarize_over_repeats(time_RxM, error_mode=error_mode)

    return {
        "x": np.array(N_grid),
        "emp_mean": emp_mean, "emp_err": emp_err,
        "est_mean": est_mean, "est_err": est_err,
        "t_mean": t_mean,     "t_err": t_err,
        "mode": mode,
    }


def sweep_over_S(S_grid, N_fixed, mode: str, error_mode="std"):
    M = len(S_grid)
    emp_RxM  = np.zeros((R, M))
    est_RxM  = np.zeros((R, M))
    time_RxM = np.zeros((R, M))

    for idx in tqdm(range(R * M), desc=f"Sweep over S [{mode}]"):
        r = idx // M
        j = idx % M
        Nsamples = S_grid[j]
        seed = BASE_SEED + 10_000 + r*1000 + j
        seed_everything(seed)

        sigma_emp, sigma_est, t_est = get_or_run(
            N=N_fixed, samples_per_spin=Nsamples, beta=beta, k=k,
            seed=seed, mode=mode, overwrite=OVERWRITE
        )
        emp_RxM[r, j]  = sigma_emp
        est_RxM[r, j]  = sigma_est
        time_RxM[r, j] = t_est

    emp_mean, emp_err = summarize_over_repeats(emp_RxM,  error_mode=error_mode)
    est_mean, est_err = summarize_over_repeats(est_RxM,  error_mode=error_mode)
    t_mean,   t_err   = summarize_over_repeats(time_RxM, error_mode=error_mode)

    return {
        "x": np.array(S_grid),
        "emp_mean": emp_mean, "emp_err": emp_err,
        "est_mean": est_mean, "est_err": est_err,
        "t_mean": t_mean,     "t_err": t_err,
        "mode": mode,
    }


# ============================================================
# Dual plotting helpers (plot mp & nmp together)
# ============================================================
def plot_dual_over_N(res_mp, res_nmp, fname_suffix=""):
    setup_matplotlib_style()
    cmap = plt.get_cmap('inferno_r')
    c_mp  = cmap(0.75)
    c_nmp = cmap(0.5)

    # EP vs N (both)
    fig, ax = plt.subplots(figsize=(4, 4))
    # Empirical (per grid)
    ax.plot(res_mp["x"],  res_mp["emp_mean"],  'k', linestyle=(0, (2, 3)), lw=2, label=r'$\Sigma$ (emp, mp grid)')
    ax.plot(res_nmp["x"], res_nmp["emp_mean"], 'k', linestyle=(0, (1, 2)), lw=2, label=r'$\Sigma$ (emp, nmp grid)')
    # Estimators
    plot_series(ax, res_mp["x"],  res_mp["est_mean"],  res_mp["est_err"],  r'multipartite', color=c_mp,  lw=2, fill=(R>1))
    plot_series(ax, res_nmp["x"], res_nmp["est_mean"], res_nmp["est_err"], r'regular',      color=c_nmp, lw=2, fill=(R>1))
    ax.set_xlim([min(res_mp["x"][0], res_nmp["x"][0]), max(res_mp["x"][-1], res_nmp["x"][-1])])
    ymax = 1.05 * max(np.max(res_mp["emp_mean"]), np.max(res_nmp["emp_mean"]),
                      np.max(res_mp["est_mean"]), np.max(res_nmp["est_mean"]))
    ax.set_ylim([0, ymax])
    style_axes(ax, r'Number of spins', 'EP')
    plt.savefig(os.path.join(IMG_DIR, f'Fig_N_EP{fname_suffix}.pdf'), bbox_inches='tight', pad_inches=0.1)

    # Time vs N (both)
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_series(ax, res_mp["x"],  res_mp["t_mean"],  res_mp["t_err"],  r'multipartite', color=c_mp,  lw=2, fill=(R>1))
    plot_series(ax, res_nmp["x"], res_nmp["t_mean"], res_nmp["t_err"], r'regular',      color=c_nmp, lw=2, fill=(R>1))
    ax.set_xlim([min(res_mp["x"][0], res_nmp["x"][0]), max(res_mp["x"][-1], res_nmp["x"][-1])])
    ymax_t = 1.05 * max(np.max(res_mp["t_mean"]), np.max(res_nmp["t_mean"]))
    ax.set_ylim([0, ymax_t])
    style_axes(ax, r'Number of spins', 'Time (s)', ypad=12)
    plt.savefig(os.path.join(IMG_DIR, f'Fig_N_Time{fname_suffix}.pdf'), bbox_inches='tight', pad_inches=0.1)


def plot_dual_over_S(res_mp, res_nmp, fname_suffix=""):
    setup_matplotlib_style()
    cmap = plt.get_cmap('inferno_r')
    c_mp  = cmap(0.75)
    c_nmp = cmap(0.5)

    # EP vs S (both)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(res_mp["x"],  res_mp["emp_mean"],  'k', linestyle=(0, (2, 3)), lw=2, label=r'$\Sigma$ (emp, mp grid)')
    ax.plot(res_nmp["x"], res_nmp["emp_mean"], 'k', linestyle=(0, (1, 2)), lw=2, label=r'$\Sigma$ (emp, nmp grid)')
    plot_series(ax, res_mp["x"],  res_mp["est_mean"],  res_mp["est_err"],  r'multipartite', color=c_mp,  lw=2, fill=(R>1))
    plot_series(ax, res_nmp["x"], res_nmp["est_mean"], res_nmp["est_err"], r'regular',      color=c_nmp, lw=2, fill=(R>1))
    ax.set_xlim([min(res_mp["x"][0], res_nmp["x"][0]), max(res_mp["x"][-1], res_nmp["x"][-1])])
    ymax = 1.05 * max(np.max(res_mp["emp_mean"]), np.max(res_nmp["emp_mean"]),
                      np.max(res_mp["est_mean"]), np.max(res_nmp["est_mean"]))
    ax.set_ylim([0, ymax])
    style_axes(ax, r'Number of samples', 'EP')
    # move the offset text
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    off = ax.xaxis.get_offset_text()
    off.set_x(1.1)     # push to the right (axes coords)
#    off.set_y(-0.12)    # push downward; tweak (-0.08 .. -0.2)
    plt.savefig(os.path.join(IMG_DIR, f'Fig_S_EP{fname_suffix}.pdf'), bbox_inches='tight', pad_inches=0.1)

    # Time vs S (both)
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_series(ax, res_mp["x"],  res_mp["t_mean"],  res_mp["t_err"],  r'multipartite', color=c_mp,  lw=2, fill=(R>1))
    plot_series(ax, res_nmp["x"], res_nmp["t_mean"], res_nmp["t_err"], r'regular',      color=c_nmp, lw=2, fill=(R>1))
    ax.set_xlim([min(res_mp["x"][0], res_nmp["x"][0]), max(res_mp["x"][-1], res_nmp["x"][-1])])
    ymax_t = 1.05 * max(np.max(res_mp["t_mean"]), np.max(res_nmp["t_mean"]))
    ax.set_ylim([0, ymax_t])
    style_axes(ax, r'Number of samples', 'Time (s)', ypad=12, legend=True)
    # move the offset text
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    off = ax.xaxis.get_offset_text()
    off.set_x(1.1)     # push to the right (axes coords)
#    off.set_y(-0.12)    # push downward; tweak (-0.08 .. -0.2)
    plt.savefig(os.path.join(IMG_DIR, f'Fig_S_Time{fname_suffix}.pdf'), bbox_inches='tight', pad_inches=0.1)



# ============================================================
# Run sweeps
# ============================================================
samples_per_spin_fixed = 20000
ERROR_MODE = "std"   # "sem" or "std"

# N-sweeps (mp & nmp)
res_N_mp  = sweep_over_N(N_mp,  samples_per_spin_fixed, mode=MP_MODE,  error_mode=ERROR_MODE)
res_N_nmp = sweep_over_N(N_nmp, samples_per_spin_fixed, mode=NMP_MODE, error_mode=ERROR_MODE)

# S-sweeps (mp & nmp) – both at the same N_for_Ss (as requested)
res_S_mp  = sweep_over_S(S_mp,  N_fixed=N_for_Ss, mode=MP_MODE,  error_mode=ERROR_MODE)
res_S_nmp = sweep_over_S(S_nmp, N_fixed=N_for_Ss, mode=NMP_MODE, error_mode=ERROR_MODE)

# Dual plots (both curves together)
plot_dual_over_N(res_N_mp,  res_N_nmp)
plot_dual_over_S(res_S_mp,  res_S_nmp)

plt.show()

