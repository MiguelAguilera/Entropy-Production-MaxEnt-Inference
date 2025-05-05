import numpy as np
from tqdm import tqdm
import re
import h5py
import hdf5plugin # This needs to be imported even thought its not explicitly used
import time, os, sys, gc
from threading import Thread

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]='1'
import torch

sys.path.insert(0, '..')
from ep_estimators import EPEstimators
import utils


# -------------------------------
# Entropy Production Calculation Functions
# -------------------------------

def get_spin_data(i, file_name, cap=None):
    """
    Loads data for spin `i` from a .npz file.
    Returns capped spin trajectory, coupling matrix row, and number of flips.
    """
    data = np.load(file_name)
    F_i = data["F"][:, i]
    S_i = data["S_bin"][F_i, :]
    J_i = data["J"][i, :]
    nflips = S_i.shape[0]
    
    if cap is not None:
        S_i = S_i[:cap, :]
        
    return S_i, J_i, nflips

        
def load_results_from_file(file_name_out, N, return_parameters=False):
    """
    Loads precomputed entropy production results from an HDF5 file.
    Returns entropy values and optionally theta parameter arrays.
    """
    S_Emp = S_TUR = S_Gaussian = S_MaxEnt = time_tur = time_Gaussian = time_MaxEnt = 0
    theta_Gaussian_list, theta_MaxEnt_list = [], []

    with h5py.File(file_name_out, 'r') as f_out:
        for i in range(N):
            if f"spins/{i}/Emp" not in f_out:
                continue
            group = f_out[f"spins/{i}"]
            S_Emp += group["Emp"][()]
            S_TUR += group["MTUR"][()]
            S_Gaussian += group["N1"][()]
            S_MaxEnt += group["N2"][()]
            time_tur += group["time_tur"][()]
            time_Gaussian += group["time_Gaussian"][()]
            time_MaxEnt += group["time_MaxEnt"][()]
            if return_parameters:
                theta_Gaussian_list.append(group["theta_Gaussian"][:])
                theta_MaxEnt_list.append(group["theta_MaxEnt"][:])

    if return_parameters:
        return (
            S_Emp, S_TUR, S_Gaussian, S_MaxEnt,
            time_tur, time_Gaussian, time_MaxEnt,
            np.array(theta_Gaussian_list), np.array(theta_MaxEnt_list)
        )
    return S_Emp, S_TUR, S_Gaussian, S_MaxEnt, time_tur, time_Gaussian, time_MaxEnt

def get_g_observables(S_i, i):
    g_samples = -2 * np.einsum('i,ij->ij', S_i[:, i], S_i)
    # We remove the i-th observable because its always 1
    g_samples = np.hstack([g_samples [:,:i], g_samples [:,i+1:]])
    return g_samples
    
    
def calc_spin(i_args):
    """
    Performs entropy production estimation for a single spin using multiple methods.
    Saves results in the specified HDF5 file.
    """
    i, N, beta, rep, T, file_name, file_name_out, S_i_t, J_i_t, nflips = i_args

    if S_i_t.shape[0] <= 100:
        print(f"[Warning] Skipping spin {i}: insufficient spin flips")
        return None

    Pi = nflips / T
    
    mask = torch.ones(S_i_t.shape[1], dtype=bool)
    mask[i] = False
    g_samples = -2 * S_i_t[:, i][:, None] * S_i_t[:,mask]

    g_mean      = g_samples.mean(axis=0)
    
    g_mean_ford_plus_back = g_mean.clone()
    g_cov_ford_minus_back = (g_samples.T @ g_samples) / g_samples.shape[0]

    J_without_i = torch.cat((J_i_t[:i], J_i_t[i+1:]))

    # Calculate empirical estimate of true EP
    spin_emp  = beta * J_without_i @ g_mean
    
    ep_estimator = EPEstimators(g_mean=g_mean, rev_g_samples=-g_samples,g_mean_ford_plus_back=g_mean_ford_plus_back, g_cov_ford_minus_back=g_cov_ford_minus_back, num_chunks=5)

    # Compute MTUR
    t0 = time.time()
    sig_MTUR, _, _ = ep_estimator.get_EP_MTUR()
    MTUR = Pi * sig_MTUR
    time_tur = time.time() - t0

    # Compute 1-step Newton
    t0 = time.time()
    sig_Gaussian, theta_Gaussian, _ = ep_estimator.get_EP_Newton(max_iter=1, holdout=True, adjust_radius=True)
    N1 = Pi * sig_Gaussian
    theta_Gaussian_np = theta_Gaussian.detach().cpu().numpy()
    time_Gaussian = time.time() - t0

    # Compute empirical EP
    Emp = Pi * spin_emp

    # Compute Newton estimation
    sig_MaxEnt, theta_MaxEnt, _ = ep_estimator.get_EP_Newton(trust_radius=0.25, holdout=True, adjust_radius=False)
    N2 = Pi * sig_MaxEnt
    theta_MaxEnt_np = theta_MaxEnt.detach().cpu().numpy()
    time_MaxEnt = time.time() - t0

    # Free memory
    del S_i_t, J_i_t
    torch.cuda.empty_cache()
    gc.collect()

    # Save results to file
    with h5py.File(file_name_out, 'a') as f_out:
        spin_group = f"spins/{i}"
        if spin_group in f_out:
            del f_out[spin_group]
        group = f_out.create_group(spin_group)
        group.create_dataset("MTUR", data=MTUR)
        group.create_dataset("time_tur", data=time_tur)
        group.create_dataset("N1", data=N1)
        group.create_dataset("theta_Gaussian", data=theta_Gaussian_np)
        group.create_dataset("time_Gaussian", data=time_Gaussian)
        group.create_dataset("N2", data=N2)
        group.create_dataset("theta_MaxEnt", data=theta_MaxEnt_np)
        group.create_dataset("time_MaxEnt", data=time_MaxEnt)
        group.create_dataset("Emp", data=Emp)

    
def calc(N, beta, rep, file_name, file_name_out, return_parameters=False, overwrite=True, check_memory=True):
    """
    Compute entropy production estimates for a full spin system.

    Args:
        N (int): Number of spins in the system.
        beta (float): Inverse temperature parameter.
        rep (int): Number of repetitions (affects total flip attempts T = N × rep).
        file_name (str): Path to input .npz file with spin data.
        file_name_out (str): Path to output .h5 file.
        return_parameters (bool): Whether to return theta parameters.
        parallel (bool): Whether to use parallel processing
        num_processes (int): Number of parallel workers (unused if parallel=False).
        overwrite (bool): Whether to overwrite output file if it exists.
        check_memory (bool): Check GPU memory before allocation.

    Returns:
        np.ndarray: EP estimates for [Empirical, MTUR, Newton-1, Newton-2]
        (optionally) theta parameters and J matrix
    """

    # Early exit if result file already exists
    if os.path.exists(file_name_out) and not overwrite:
        print(f"[Info] Output file '{file_name_out}' already exists. Skipping computation.")
        results = load_results_from_file(file_name_out, N, return_parameters)
        if not return_parameters:
            S_Emp, S_TUR, S_Gaussian, S_MaxEnt, time_tur, time_Gaussian, time_MaxEnt = results
            return np.array([S_Emp, S_TUR, S_Gaussian, S_MaxEnt])
        else:
            S_Emp, S_TUR, S_Gaussian, S_MaxEnt, time_tur, time_Gaussian, time_MaxEnt, theta_Gaussian, theta_MaxEnt = results
            data = np.load(file_name)
            return np.array([S_Emp, S_TUR, S_Gaussian, S_MaxEnt]), np.array(theta_Gaussian), np.array(theta_MaxEnt), data['J']
    print()
    print("=" * 70)
    print(f"  Starting EP estimation | System size: {N} | β = {beta:.4f}")
    print("=" * 70)

    # Total spin-flip attempts
    T = N * rep
    cap = None  # Cap on number of flips per spin (disabled)

    class DummyProgress:
        def __init__(self):
            self.value = 0

    progress = DummyProgress()
    print("[Sequential] Running on a single process")

    preload_depth = 4  # Number of spins to preload in parallel
    preload_threads = {}
    preload_results = {}

    # Preload first few spins using threads
    for preload_i in range(min(preload_depth, N)):
        def preload_func(j=preload_i):
            preload_results[j] = get_spin_data(j, file_name, cap=cap)
        thread = Thread(target=preload_func)
        thread.start()
        preload_threads[preload_i] = thread

    # Temporary output file (avoid corruption on crash)
    temp_file_name_out = file_name_out + ".tmp"
    if os.path.exists(temp_file_name_out):
        os.remove(temp_file_name_out)

    # Main loop over all spins
    for i in tqdm(range(N), desc="Sequential spin progress"):
        if i in preload_threads:
            preload_threads[i].join()
            S_i, J_i, nflips = preload_results.pop(i)
            preload_threads.pop(i)
        else:
            S_i, J_i, nflips = get_spin_data(i, file_name, cap=cap)

        # Convert to torch tensors on appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        S_i_t = torch.from_numpy(S_i).to(device).float() * 2 - 1  # {0,1} → {-1,1}
        J_i_t = torch.from_numpy(J_i).to(device)

        del S_i, J_i  # Free memory

        # Perform estimation
        calc_spin((i, N, beta, rep, T, file_name, temp_file_name_out, S_i_t, J_i_t, nflips))
        progress.value += 1

        # Preload next spin in background
        next_spin = i + preload_depth
        if next_spin < N and next_spin not in preload_threads:
            def preload_func(j=next_spin):
                preload_results[j] = get_spin_data(j, file_name, cap=cap)
            thread = Thread(target=preload_func)
            thread.start()
            preload_threads[next_spin] = thread

    # Finalize: move temp file to target path
    os.rename(temp_file_name_out, file_name_out)


    # --------------------------------------------
    # Return or print results
    # --------------------------------------------
    results = load_results_from_file(file_name_out, N, return_parameters)
    if return_parameters:
        S_Emp, S_TUR, S_Gaussian, S_MaxEnt, time_tur, time_Gaussian, time_MaxEnt, theta_Gaussian, theta_MaxEnt = results
    else:
        S_Emp, S_TUR, S_Gaussian, S_MaxEnt, time_tur, time_Gaussian, time_MaxEnt = results

    print("\n[Results]")
    print(f"  Empirical     : {S_Emp:.6f}")
    print(f"  MTUR          : {S_TUR:.6f}   | Time: {time_tur:.2f} s")
    print(f"  Gaussian      : {S_Gaussian:.6f}   | Time: {time_Gaussian:.2f} s")
    print(f"  MaxEnt        : {S_MaxEnt:.6f}   | Time: {time_MaxEnt:.2f} s")
    print("-" * 70)

    if return_parameters:
        data = np.load(file_name)
        return np.array([S_Emp, S_TUR, S_Gaussian, S_MaxEnt]), theta_Gaussian, theta_MaxEnt, data["J"]
    else:
        return np.array([S_Emp, S_TUR, S_Gaussian, S_MaxEnt])
        
