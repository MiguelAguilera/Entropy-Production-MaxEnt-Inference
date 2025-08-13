import time, os, gc, random
import numpy as np
from matplotlib import pyplot as plt
import torch

import spin_model
import observables
import ep_estimators
import utils

from tqdm import tqdm

# ============================================================
# Config
# ============================================================
utils.set_default_torch_device()

# Sweep over system size
Ns = [10, 15, 20, 25, 30, 35, 40]

# Sweep over samples per spin
Ss = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]


Ss = [10000]
Ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

k = 6            # avg number of neighbors in sparse coupling matrix
beta = 2.0       # inverse temperature
N_for_Ss = 40    # N used in the S-sweep

R = 20            # <-- number of repeats per case (change as you like)
BASE_SEED = 1234 # base seed; per-repeat seeds are derived as BASE_SEED + r

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

def summarize_over_repeats(results_RxM):
    """
    results_RxM: np.array shape (R, M)
    Returns mean (M,), std (M,)
    """
    mean = results_RxM.mean(axis=0)
    std = results_RxM.std(axis=0, ddof=1) if results_RxM.shape[0] > 1 else np.zeros(results_RxM.shape[1])
    return mean, std

def plot_with_fill(x, y_mean, y_std, label, use_fill):
    plt.plot(x, y_mean, 'o-', label=label)
    if use_fill and np.any(y_std > 0):
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)

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
    seed_everything(BASE_SEED + r*1000 + j)

    sigma_emp, sigma_mp, sigma_nmp, t_mp, t_nmp = run_single_experiment(
        N=N, samples_per_spin=samples_per_spin_fixed, beta=beta, k=k
    )
    emp_RxM[r, j]  = sigma_emp
    mp_RxM[r, j]   = sigma_mp
    nmp_RxM[r, j]  = sigma_nmp
    tmp_RxM[r, j]  = t_mp
    tnmp_RxM[r, j] = t_nmp

emp_mean, emp_std   = summarize_over_repeats(emp_RxM)
mp_mean, mp_std     = summarize_over_repeats(mp_RxM)
nmp_mean, nmp_std   = summarize_over_repeats(nmp_RxM)
tmp_mean, tmp_std   = summarize_over_repeats(tmp_RxM)
tnmp_mean, tnmp_std = summarize_over_repeats(tnmp_RxM)

plt.rc('text', usetex=True)
plt.rc('font', size=22, family='serif', serif=['latin modern roman'])
plt.rc('legend', fontsize=20)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')


plt.figure(figsize=(10, 6))
plot_with_fill(Ns, emp_mean,  emp_std,  'Empirical EP',              use_fill=(R > 1))
plot_with_fill(Ns, mp_mean,   mp_std,   'Multipartite EP',           use_fill=(R > 1))
plot_with_fill(Ns, nmp_mean,  nmp_std,  'Non-multipartite EP',       use_fill=(R > 1))
plt.xlabel('Number of spins (N)')
plt.ylabel(r'Entropy production ($\Sigma$)')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 6))
plot_with_fill(Ns, tmp_mean,   tmp_std,   'Multipartite EP Time',     use_fill=(R > 1))
plot_with_fill(Ns, tnmp_mean,  tnmp_std,  'Non-multipartite EP Time', use_fill=(R > 1))
plt.xlabel('Number of spins (N)')
plt.ylabel('Time (seconds)')
plt.legend()
plt.tight_layout()

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
    seed_everything(BASE_SEED + 10_000 + r*1000 + j)
    sigma_emp, sigma_mp, sigma_nmp, t_mp, t_nmp = run_single_experiment(
        N=N_for_Ss, samples_per_spin=Nsamples, beta=beta, k=k
    )
    emp_RxM[r, j]  = sigma_emp
    mp_RxM[r, j]   = sigma_mp
    nmp_RxM[r, j]  = sigma_nmp
    tmp_RxM[r, j]  = t_mp
    tnmp_RxM[r, j] = t_nmp

emp_mean, emp_std   = summarize_over_repeats(emp_RxM)
mp_mean, mp_std     = summarize_over_repeats(mp_RxM)
nmp_mean, nmp_std   = summarize_over_repeats(nmp_RxM)
tmp_mean, tmp_std   = summarize_over_repeats(tmp_RxM)
tnmp_mean, tnmp_std = summarize_over_repeats(tnmp_RxM)

plt.figure(figsize=(10, 6))
plot_with_fill(Ss, emp_mean,  emp_std,  'Empirical EP',              use_fill=(R > 1))
plot_with_fill(Ss, mp_mean,   mp_std,   'Multipartite EP',           use_fill=(R > 1))
plot_with_fill(Ss, nmp_mean,  nmp_std,  'Non-multipartite EP',       use_fill=(R > 1))
plt.xlabel('Number of samples per spin (S)')
plt.ylabel(r'Entropy production ($\Sigma$)')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 6))
plot_with_fill(Ss, tmp_mean,   tmp_std,   'Multipartite EP Time',     use_fill=(R > 1))
plot_with_fill(Ss, tnmp_mean,  tnmp_std,  'Non-multipartite EP Time', use_fill=(R > 1))
plt.xlabel('Number of samples per spin (S)')
plt.ylabel('Time (seconds)')
plt.legend()
plt.tight_layout()

plt.show()
