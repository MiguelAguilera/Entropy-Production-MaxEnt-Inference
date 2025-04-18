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

def calc_spin(i_args):
    i, N, T, file_name, temp_dir = i_args
    with h5py.File(file_name, 'r') as f:
        S_i = f[f'S_{i}'][:].astype('float32') * 2 - 1
        J_i = f['J'][i, :]  # Load just the i-th row
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
    sig_Adam, theta_Adam = get_EP_Adam(S_i_t, theta_N1, Da, i)
    result["Adam"] = Pi * sig_Adam
    result["time_adam"] = time.time() - t0
    
    result["Emp"] = Pi * exp_EP_spin_model(Da, J_i_t, i)
    
    # Save parameter vectors to temporary HDF5 file
    file_uuid = f"theta_vectors_spin_{i}_{uuid.uuid4().hex}.h5"
    path = os.path.join(temp_dir, file_uuid)
    with h5py.File(path, 'w') as f:
        f.create_dataset("theta_N1", data=theta_N1.detach().cpu().numpy())
        f.create_dataset("theta_N2", data=theta_N2.detach().cpu().numpy())
        f.create_dataset("theta_Adam", data=theta_Adam.detach().cpu().numpy())

    result["theta_file"] = path
    return result
    
def calc(N, rep, file_name, return_parameters=False):
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

    with tempfile.TemporaryDirectory() as temp_dir:
        args_list = [(i, N, T, file_name, temp_dir) for i in range(N)]
#        with mp.Pool(processes=mp.cpu_count()) as pool:
#            results = pool.map(calc_spin, args_list)

#        results = Parallel(n_jobs=mp.cpu_count())(
#            delayed(calc_spin)(args) for args in args_list
#        )
        results = Parallel(
            n_jobs=mp.cpu_count(),
            backend="multiprocessing",  # Or "threading" if I/O bound
            prefer="processes"
        )(
            delayed(calc_spin)(args) for args in args_list
        )
        
        # Aggregate scalar values
        S_Emp, S_TUR, S_N1, S_N2, S_Adam, time_tur, time_n1, time_n2, time_adam = map(
            sum, zip(*[(r["Emp"], r["MTUR"], r["N1"], r["N2"], r["Adam"],
                        r["time_tur"], r["time_n1"], r["time_n2"], r["time_adam"])
                       for r in results if r is not None])
        )

        print("\n[Results]")
        print(f"  Empirical       : {S_Emp:.6f}")
        print(f"  MTUR            : {S_TUR:.6f}   | Time: {time_tur:.2f} s")
        print(f"  1-step Newton   : {S_N1:.6f}   | Time: {time_n1:.2f} s")
        print(f"  2-step Newton   : {S_N2:.6f}   | Time: {time_n2:.2f} s")
        print(f"  Adam            : {S_Adam:.6f}   | Time: {time_adam:.2f} s")
        print("-" * 70)

        if return_parameters==False:
            return np.array([S_Emp, S_TUR, S_N1, S_N2,S_Adam])
        else:
            theta_N1_list = []
            theta_N2_list = []
            theta_Adam_list = []

            for r in results:
                if r is None:
                    continue
                with h5py.File(r["theta_file"], 'r') as f:
                    theta_N1_list.append(f["theta_N1"][:])
                    theta_N2_list.append(f["theta_N2"][:])
                    theta_Adam_list.append(f["theta_Adam"][:])

            theta_N1 = np.array(theta_N1_list)  # Shape: (N_effective, D)
            theta_N2 = np.array(theta_N2_list)
            theta_Adam = np.array(theta_Adam_list)

            print(f"  [Debug] theta_N1 shape: {theta_N1.shape}")
            print(f"  [Debug] theta_N2 shape: {theta_N2.shape}")
            print(f"  [Debug] theta_Adam shape: {theta_Adam.shape}")
            
            return np.array([S_Emp, S_TUR, S_N1, S_N2,S_Adam]),theta_N1,theta_N2,theta_Adam, J
