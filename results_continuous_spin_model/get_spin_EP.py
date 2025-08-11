import os
import sys
import gc
import time
from threading import Thread

import numpy as np
import torch
import h5py
import hdf5plugin  # keep for HDF5 compression backends
from tqdm import tqdm

# Set PyTorch environment variables
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Extend path to find custom modules
sys.path.insert(0, '..')
import ep_estimators
import observables  # only for Dataset


# -------------------------------
# Data loading for continuous spins
# -------------------------------

def get_spin_data(i, file_name, cap=None):
    """
    Load trajectories for spin i and build g_observables inline.

    Returns:
        X  : (T_i, N-1) array with entries cos(S1_i - S_j) - cos(S_i - S_j), j != i
        Ji : (N,) coupling vector for spin i
        nsteps : int, number of time steps used
    """
    data = np.load(file_name)
    # Continuous model stores angles (radians)
    S  = data["S"]    # (T, N)
    S1 = data["S1"]   # (T, N)
    Ji = data["J"][i, :]  # (N,)

    if cap is not None:
        S  = S[:cap, :]
        S1 = S1[:cap, :]

    # ---- Inline g-observables (no call to observables.get_g_observables) ----
    # dC(t, j) = cos(S1_i(t) - S_j(t)) - cos(S_i(t) - S_j(t))
    dC = np.cos(S1[:, [i]] - S) - np.cos(S[:, [i]] - S)   # (T, N)
    # drop self column j=i
    X = np.hstack((dC[:, :i], dC[:, i+1:]))               # (T, N-1)
    # -------------------------------------------------------------------------

    nsteps = X.shape[0]
    return X, Ji, nsteps


# -------------------------------
# Estimation Core Function
# -------------------------------

def calc_spin(i_args):
    """
    Performs entropy production estimation for a single spin and saves to HDF5.
    """
    (i, N, beta, rep, T, file_name, file_name_out,
     g_samples, J_i_t, nsteps, seed) = i_args

    if g_samples.shape[0] <= 100:
        print(f"[Warning] Skipping spin {i}: insufficient samples ({g_samples.shape[0]})")
        return

    # Each spin's weight ≈ attempts_i / total_attempts
    Pi = nsteps / T

    # Remove self-coupling (consistent with previous convention)
    J_without_i = torch.cat((J_i_t[:i], J_i_t[i+1:]))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Empirical EP contribution for spin i
    g_mean = torch.mean(g_samples, axis=0)
    spin_emp = beta * (J_without_i @ g_mean).item()

    # Prepare dataset for learned estimators
    data = observables.Dataset(g_samples=g_samples, num_chunks=5)
    trn, val, tst = data.split_train_val_test(val_fraction=0.2, test_fraction=0.1)

    # MTUR
    t0 = time.time()
    sig_MTUR, _ = ep_estimators.get_EP_MTUR(data)
    MTUR = Pi * sig_MTUR
    time_tur = time.time() - t0

    # One-step Newton
    t0 = time.time()
    sig_hat_g, theta_hat_g = ep_estimators.get_EP_Newton1Step(trn, validation=val, test=tst)
    EP_hat_g = Pi * sig_hat_g
    time_hat_g = time.time() - t0

    # Gradient descent (BB)
    t0 = time.time()
    optimizer_kwargs = dict(lr=0.001, patience=10, tol=1e-8)
    sig_g, theta_g = ep_estimators.get_EP_Estimate(
        trn, validation=val, test=tst,
        optimizer='GradientDescentBB', optimizer_kwargs=optimizer_kwargs
    )
    EP_g = Pi * sig_g
    time_g = time.time() - t0

    Emp = Pi * spin_emp

    # Free memory
    del g_samples, J_i_t
    torch.cuda.empty_cache()
    gc.collect()

    # Save results
    with h5py.File(file_name_out, 'a') as f_out:
        spin_group = f"spins/{i}"
        if spin_group in f_out:
            del f_out[spin_group]
        group = f_out.create_group(spin_group)
        group.create_dataset("MTUR", data=MTUR)
        group.create_dataset("time_tur", data=time_tur)
        group.create_dataset("EP_hat_g", data=EP_hat_g)
        group.create_dataset("theta_hat_g", data=theta_hat_g)
        group.create_dataset("time_hat_g", data=time_hat_g)
        group.create_dataset("EP_g", data=EP_g)
        group.create_dataset("theta_g", data=theta_g)
        group.create_dataset("time_g", data=time_g)
        group.create_dataset("Emp", data=Emp)


# -------------------------------
# Full System Evaluation
# -------------------------------

def load_results_from_file(file_name_out, N, return_parameters=False):
    S_Emp = S_TUR = S_hat_g = S_g = 0
    time_tur = time_hat_g = time_g = 0
    theta_hat_g_list, theta_g_list = [], []

    with h5py.File(file_name_out, 'r') as f_out:
        for i in range(N):
            if f"spins/{i}/Emp" not in f_out:
                continue
            group = f_out[f"spins/{i}"]
            S_Emp += group["Emp"][()]
            S_TUR += group["MTUR"][()]
            S_hat_g += group["EP_hat_g"][()]
            S_g += group["EP_g"][()]
            time_tur += group["time_tur"][()]
            time_hat_g += group["time_hat_g"][()]
            time_g += group["time_g"][()]
            if return_parameters:
                theta_hat_g_list.append(group["theta_hat_g"][:])
                theta_g_list.append(group["theta_g"][:])

    if return_parameters:
        return (
            S_Emp, S_TUR, S_hat_g, S_g,
            time_tur, time_hat_g, time_g,
            np.array(theta_hat_g_list), np.array(theta_g_list)
        )
    return S_Emp, S_TUR, S_hat_g, S_g, time_tur, time_hat_g, time_g


def calc(N, beta, rep, file_name, file_name_out, return_parameters=False,
         overwrite=True, check_memory=True, seed=None):
    """
    Computes EP estimates for all spins using S,S1 (continuous spins).
    file_name must contain arrays: S, S1, J.
    """
    if os.path.exists(file_name_out) and not overwrite:
        print(f"[Info] Output file '{file_name_out}' exists. Skipping.")
        results = load_results_from_file(file_name_out, N, return_parameters)
        if return_parameters:
            S_Emp, S_TUR, S_hat_g, S_g, *_ , theta_hat_g, theta_g = results
            data = np.load(file_name)
            return np.array([S_Emp, S_TUR, S_hat_g, S_g]), theta_hat_g, theta_g, data['J']
        else:
            return np.array(results[:4])

    print("\n" + "=" * 70)
    print(f"  Starting EP estimation | N={N} | β={beta:.4f}")
    print("=" * 70)

    T = N * rep
    cap = None
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
        print(f"No seed provided. Using random seed: {seed}")

    class DummyProgress:
        def __init__(self):
            self.value = 0

    progress = DummyProgress()
    print("[Sequential] Running on a single process")

    preload_depth = 5
    preload_threads = {}
    preload_results = {}

    # Preload a few spins
    for preload_i in range(min(preload_depth, N)):
        def preload_func(j=preload_i):
            preload_results[j] = get_spin_data(j, file_name, cap)
        thread = Thread(target=preload_func)
        thread.start()
        preload_threads[preload_i] = thread

    temp_file_name_out = file_name_out + ".tmp"
    if os.path.exists(temp_file_name_out):
        os.remove(temp_file_name_out)

    for i in tqdm(range(N), desc="Sequential spin progress"):
        if i in preload_threads:
            preload_threads[i].join()
            X_i, J_i, nsteps = preload_results.pop(i)
            preload_threads.pop(i)
        else:
            X_i, J_i, nsteps = get_spin_data(i, file_name, cap)

        device = (
            torch.device("mps") if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )
        # Continuous features: no rescaling
        g_samples = torch.from_numpy(X_i).to(device).float()
        J_i_t = torch.from_numpy(J_i).to(device)

        del X_i, J_i

        calc_spin((i, N, beta, rep, T, file_name, temp_file_name_out, g_samples, J_i_t, nsteps, seed))
        progress.value += 1

        next_spin = i + preload_depth
        if next_spin < N:
            def preload_func(j=next_spin):
                preload_results[j] = get_spin_data(j, file_name, cap)
            thread = Thread(target=preload_func)
            thread.start()
            preload_threads[next_spin] = thread

    os.rename(temp_file_name_out, file_name_out)

    results = load_results_from_file(file_name_out, N, return_parameters)

    if return_parameters:
        S_Emp, S_TUR, S_hat_g, S_g, time_tur, time_hat_g, time_g, theta_hat_g, theta_g = results
    else:
        S_Emp, S_TUR, S_hat_g, S_g, time_tur, time_hat_g, time_g = results

    print("\n[Results]")
    print(f"  Empirical     : {S_Emp:.6f}")
    print(f"  MTUR          : {S_TUR:.6f}   | Time: {time_tur:.2f} s")
    print(f"  Gaussian      : {S_hat_g:.6f} | Time: {time_hat_g:.2f} s")
    print(f"  MaxEnt        : {S_g:.6f}     | Time: {time_g:.2f} s")
    print("-" * 70)

    if return_parameters:
        data = np.load(file_name)
        return np.array([S_Emp, S_TUR, S_hat_g, S_g]), theta_hat_g, theta_g, data["J"]
    else:
        return np.array(results[:4])

