import os, sys, time, gc, re
import numpy as np
import torch
import h5py
import hdf5plugin  # Required to read compressed HDF5 files (even if unused)
from threading import Thread
from tqdm import tqdm

import utils

sys.path.insert(0, '..')
from ep_multipartite import EPEstimators

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
    S_Emp = S_TUR = S_N1 = S_N2 = time_tur = time_n1 = time_n2 = 0
    theta_N1_list, theta_N2_list = [], []

    with h5py.File(file_name_out, 'r') as f_out:
        for i in range(N):
            if f"spins/{i}/Emp" not in f_out:
                continue
            group = f_out[f"spins/{i}"]
            S_Emp += group["Emp"][()]
            S_TUR += group["MTUR"][()]
            S_N1 += group["N1"][()]
            S_N2 += group["N2"][()]
            time_tur += group["time_tur"][()]
            time_n1 += group["time_n1"][()]
            time_n2 += group["time_n2"][()]
            if return_parameters:
                theta_N1_list.append(group["theta_N1"][:])
                theta_N2_list.append(group["theta_N2"][:])

    if return_parameters:
        return (
            S_Emp, S_TUR, S_N1, S_N2,
            time_tur, time_n1, time_n2,
            np.array(theta_N1_list), np.array(theta_N2_list)
        )
    return S_Emp, S_TUR, S_N1, S_N2, time_tur, time_n1, time_n2

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
    ep_estimator = EPEstimators(S_i_t, i, num_chunks=5)

    # Compute MTUR
    t0 = time.time()
    sig_MTUR, _, _ = ep_estimator.get_EP_MTUR()
    MTUR = Pi * sig_MTUR
    time_tur = time.time() - t0

    # Compute 1-step Newton
    t0 = time.time()
    sig_N1, theta_N1, _ = ep_estimator.get_EP_Newton(max_iter=1, holdout=True, adjust_radius=True)
    N1 = Pi * sig_N1
    theta_N1_np = theta_N1.detach().cpu().numpy()
    time_n1 = time.time() - t0

    # Compute empirical EP
    Emp = Pi * beta * float(utils.remove_i(J_i_t, i) @ ep_estimator.g_mean())

    # Compute 2-step Newton
    sig_N2, theta_N2, _ = ep_estimator.get_EP_Newton(trust_radius=0.25, holdout=True, adjust_radius=False)
    N2 = Pi * sig_N2
    theta_N2_np = theta_N2.detach().cpu().numpy()
    time_n2 = time.time() - t0

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
        group.create_dataset("theta_N1", data=theta_N1_np)
        group.create_dataset("time_n1", data=time_n1)
        group.create_dataset("N2", data=N2)
        group.create_dataset("theta_N2", data=theta_N2_np)
        group.create_dataset("time_n2", data=time_n2)
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
        return results if return_parameters else np.array(results[:4])

    # Infer beta from filename if needed
    if beta is None:
        beta_str = re.search(r'_beta_([0-9.]+)', file_name).group(1)
        beta = float(beta_str)

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
    # 🧾 Return or print results
    # --------------------------------------------
    results = load_results_from_file(file_name_out, N, return_parameters)
    if return_parameters:
        S_Emp, S_TUR, S_N1, S_N2, time_tur, time_n1, time_n2, theta_N1, theta_N2 = results
    else:
        S_Emp, S_TUR, S_N1, S_N2, time_tur, time_n1, time_n2 = results

    print("\n[Results]")
    print(f"  Empirical       : {S_Emp:.6f}")
    print(f"  MTUR            : {S_TUR:.6f}   | Time: {time_tur:.2f} s")
    print(f"  1-step Newton   : {S_N1:.6f}   | Time: {time_n1:.2f} s")
    print(f"  2-step Newton   : {S_N2:.6f}   | Time: {time_n2:.2f} s")
    print("-" * 70)

    if return_parameters:
        data = np.load(file_name)
        return np.array([S_Emp, S_TUR, S_N1, S_N2]), theta_N1, theta_N2, data["J"]
    else:
        return np.array([S_Emp, S_TUR, S_N1, S_N2])

