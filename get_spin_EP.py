import torch
import numpy as np
import multiprocessing as mp
import re
import h5py
import hdf5plugin # This needs to be imported even thought its not explicitly used
import time
import os
import uuid, tempfile
from joblib import Parallel, delayed
from methods_EP_multipartite import *


# -------------------------------
# Main Entropy Production Calculation Function
# -------------------------------


def get_spin_data(i, file_name):
    with h5py.File(file_name, 'r') as f:
        F_i = f['F'][i]             # loads only 1 row of F
        S_i = f['S'][:, F_i].astype('float32') * 2 - 1  # convert back to {-1, 1}
        J_i = f['J'][i, :]          # loads only 1 row of J
        return S_i, J_i

def calc_spin(i_args):
    i, N, T, file_name, file_name_out, lock = i_args
    S_i, J_i = get_spin_data(i, file_name)

    S_i_t = torch.from_numpy(S_i)
    J_i_t = torch.from_numpy(J_i)

    if S_i.shape[1] <= 10:
        print(f"  [Warning] Skipping spin {i}: insufficient time steps")
        return None

    Pi = S_i.shape[1] / T

    result = {}

    t0 = time.time()
    result["MTUR"] = Pi * get_EP_MTUR(S_i_t, i)
    result["time_tur"] = time.time() - t0

    t0 = time.time()
    sig_N1, theta_N1, Da = get_EP_Newton(S_i_t, i)
    result["N1"] = Pi * sig_N1
    result["time_n1"] = time.time() - t0

    sig_N2, theta_N2 = get_EP_Newton2(S_i_t, theta_N1, Da, i)
    result["N2"] = Pi * sig_N2
    result["time_n2"] = time.time() - t0

    t0 = time.time()
#    sig_Adam, theta_Adam = get_EP_Adam(S_i_t, theta_N1, Da, i)
#    result["Adam"] = Pi * sig_Adam
#    result["time_adam"] = time.time() - t0
    
    result["Emp"] = Pi * exp_EP_spin_model(Da, J_i_t, i)
    
    # Save values to HDF5 file
    with lock:
        with h5py.File(file_name_out, 'a') as f_out:
            for name, value in result.items():
                dataset_name = f"{name}_{i}"
                if dataset_name in f_out:
                    del f_out[dataset_name]
                f_out.create_dataset(dataset_name, data=value)



def calc_spin_group(group_args):
    indices, N, T, file_name, file_name_out, lock = group_args
    results = []
    for i in indices:
        res = calc_spin((i, N, T, file_name, file_name_out, lock))
        results.append((i, res))
    
    
def load_results_from_file(file_name_out, N, return_parameters=False):
    S_Emp = S_TUR = S_N1 = S_N2 = time_tur = time_n1 = time_n2 = 0
    theta_N1_list = []
    theta_N2_list = []

    with h5py.File(file_name_out, 'r') as f_out:
        for i in range(N):
            if f"Emp_{i}" not in f_out:
                continue
            S_Emp += f_out[f"Emp_{i}"][()]
            S_TUR += f_out[f"MTUR_{i}"][()]
            S_N1 += f_out[f"N1_{i}"][()]
            S_N2 += f_out[f"N2_{i}"][()]
            time_tur += f_out[f"time_tur_{i}"][()]
            time_n1 += f_out[f"time_n1_{i}"][()]
            time_n2 += f_out[f"time_n2_{i}"][()]
            if return_parameters:
                theta_N1_list.append(f_out[f"theta_N1_{i}"][:])
                theta_N2_list.append(f_out[f"theta_N2_{i}"][:])

    if not return_parameters:
        return S_Emp, S_TUR, S_N1, S_N2, time_tur, time_n1, time_n2
    else:
        return (
            S_Emp, S_TUR, S_N1, S_N2,
            time_tur, time_n1, time_n2,
            np.array(theta_N1_list), np.array(theta_N2_list)
        )
    
def calc(N, rep, file_name, file_name_out, return_parameters=False):
    """
    Compute entropy production rate (EP) estimates using multiple methods for a spin system.

    Parameters:
        N (int): System size.
        rep (int): Number of repetitions.

    Returns:
        np.ndarray: EP estimates [empirical, MTUR, Newton-1, Newton-2]
    """
    
    beta_str = re.search(r'_beta_([0-9.]+)', file_name).group(1)
    beta = float(beta_str)
    print()
    print("=" * 70)
    print(f"  Starting EP estimation | System size: {N} | β = {beta:.4f}")
    print("=" * 70)

    with h5py.File(file_name, 'r') as f:
        J = f['J'][:]
        H = f['H'][:]
        assert(np.all(H==0))  # We do not support local fields in our analysis

    # Initialize accumulators
    S_Emp = S_TUR = S_N1 = S_N2 = S_Adam = 0
    time_emp = time_tur = time_n1 = time_n2 = time_adam = 0
    T = N * rep  # Total spin-flip attempts

    # Parallel processing

#        args_list = [(i, N, T, file_name, temp_dir) for i in range(N)]
##        with mp.Pool(processes=mp.cpu_count()) as pool:
##            results = pool.map(calc_spin, args_list)

##        results = Parallel(n_jobs=mp.cpu_count())(
##            delayed(calc_spin)(args) for args in args_list
##        )
#        results = Parallel(
#            n_jobs=mp.cpu_count(),
#            backend="multiprocessing",  # Or "threading" if I/O bound
#            prefer="processes"
#        )(
#            delayed(calc_spin)(args) for args in args_list
#        )
    lock = mp.Manager().Lock()
    num_processes = mp.cpu_count()
    group_size = int(np.ceil(N / num_processes))
    grouped_indices = [list(range(i, min(i + group_size, N))) for i in range(0, N, group_size)]
    args_list = [(indices, N, T, file_name, file_name_out, lock) for indices in grouped_indices]

    Parallel(
        n_jobs=num_processes,
        backend="multiprocessing",
        prefer="processes"
    )(delayed(calc_spin_group)(args) for args in args_list)

    # Aggregate results from HDF5
    results = load_results_from_file(file_name_out, N, return_parameters)

    if not return_parameters:
        S_Emp, S_TUR, S_N1, S_N2, time_tur, time_n1, time_n2 = results
    else:
        S_Emp, S_TUR, S_N1, S_N2, time_tur, time_n1, time_n2, theta_N1, theta_N2 = results


    print("\n[Results]")
    print(f"  Empirical       : {S_Emp:.6f}")
    print(f"  MTUR            : {S_TUR:.6f}   | Time: {time_tur:.2f} s")
    print(f"  1-step Newton   : {S_N1:.6f}   | Time: {time_n1:.2f} s")
    print(f"  2-step Newton   : {S_N2:.6f}   | Time: {time_n2:.2f} s")
    print("-" * 70)

    if not return_parameters:
        return np.array([S_Emp, S_TUR, S_N1, S_N2])
    else:
        theta_N1 = np.array(theta_N1_list)
        theta_N2 = np.array(theta_N2_list)
        return np.array([S_Emp, S_TUR, S_N1, S_N2]), theta_N1, theta_N2, J
