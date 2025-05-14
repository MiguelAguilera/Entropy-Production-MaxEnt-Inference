import numpy as np
from tqdm import tqdm
import re
import h5py
import hdf5plugin # This needs to be imported even thought its not explicitly used
import time, os, sys, gc
from threading import Thread

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]='1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

sys.path.insert(0, '..')
import ep_estimators
import utils
from numba import njit

# -------------------------------
# Entropy Production Calculation Functions
# -------------------------------

#@njit
def compute_spin_differences(S_i, i):
    """
    S_i: np.ndarray of shape (T, N), dtype=bool
    i: index of reference spin

    Returns g = 1 if any spin differs from spin i at any time, else 0.
    """
    ref = S_i[:, i]  # shape (T,)
    X = S_i ^ ref[:, None]  # shape (T, N)
    X = np.delete(X, i, axis=1)  
    return X
    
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
    mask = np.ones(S_i.shape[1], dtype=bool)
    mask[i] = False
    # Return spin changes, J_i, and number of spin flips
    X = compute_spin_differences(S_i, i)
    return X, J_i, nflips

        
def load_results_from_file(file_name_out, N, return_parameters=False):
    """
    Loads precomputed entropy production results from an HDF5 file.
    Returns entropy values and optionally theta parameter arrays.
    """
    S_Emp = S_TUR = S_hat_g = S_g = time_tur = time_hat_g = time_g = 0
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
    i, N, beta, rep, T, file_name, file_name_out, g_samples, J_i_t, nflips, seed = i_args

    if g_samples.shape[0] <= 100:
        print(f"[Warning] Skipping spin {i}: insufficient spin flips")
        return None

    Pi = nflips / T
#    print(nflips)
    
    J_without_i = torch.cat((J_i_t[:i], J_i_t[i+1:]))

    num_chunks=5
#    num_chunks=-1
#    data = ep_estimators.Dataset(g_samples)
#    est = ep_estimators.EPEstimators(data)

    torch.manual_seed(seed)
#    print("→ Torch seed {seed}  set for CPU.")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
#        print("→ Torch seed {seed} set for CUDA.")   
         
    data = ep_estimators.Dataset(g_samples=g_samples)
    trn, val, tst = data.split_train_val_test(val_fraction=0.2, test_fraction=0.1)
        
    # Calculate empirical estimate of true EP
    
    g_mean = torch.mean(g_samples, axis=0)
    spin_emp  = beta * (J_without_i @ g_mean).item()
    # Compute MTUR
    t0 = time.time()
    sig_MTUR = ep_estimators.get_EP_MTUR(data,num_chunks=num_chunks).objective
    MTUR = Pi * sig_MTUR
    time_tur = time.time() - t0

#    torch.cuda.empty_cache()
#    gc.collect()
    
    # Compute 1-step Newton
    t0 = time.time()
    sig_hat_g, theta_hat_g, sig_hat_g_trn = ep_estimators.get_EP_Newton(trn,holdout_data=tst, validation_data=val, max_iter=1, trust_radius=None, num_chunks=num_chunks)
#    theta_init=theta_hat_g.clone() # save theta for later initialization
    EP_hat_g = Pi * sig_hat_g
    EP_hat_g_trn = Pi * sig_hat_g_trn
    theta_hat_g_np = theta_hat_g.detach().cpu().numpy()
    time_hat_g = time.time() - t0

#    torch.cuda.empty_cache()
#    gc.collect()

    # Compute Newton estimation
    sig_g, theta_g, sig_g_trn = ep_estimators.get_EP_GradientAscent(data=trn, validation_data=val,  test_data=tst, use_BB=True, verbose=0)
#    sig_g, theta_g, sig_g_trn = ep_estimators.get_EP_Newton(trn,holdout_data=tst, trust_radius=1.0, adjust_radius=False, num_chunks=num_chunks, verbose=1, tol=1e-8, patience=1)
#    
    EP_g = Pi * sig_g
    EP_g_trn = Pi * sig_g_trn
    theta_g_np = theta_g.detach().cpu().numpy()
    time_g = time.time() - t0

    # Compute empirical EP
    Emp = Pi * spin_emp

    # Free memory
    del g_samples, J_i_t
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
        group.create_dataset("EP_hat_g", data=EP_hat_g)
        group.create_dataset("EP_hat_g_trn", data=EP_hat_g_trn)
        group.create_dataset("theta_hat_g", data=theta_hat_g_np)
        group.create_dataset("time_hat_g", data=time_hat_g)
        group.create_dataset("EP_g", data=EP_g)
        group.create_dataset("EP_g_trn", data=EP_g_trn)
        group.create_dataset("theta_g", data=theta_g_np)
        group.create_dataset("time_g", data=time_g)
        group.create_dataset("Emp", data=Emp)

    
def calc(N, beta, rep, file_name, file_name_out, return_parameters=False, overwrite=True, check_memory=True, seed=None):
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
            S_Emp, S_TUR, S_hat_g, S_g, time_tur, time_hat_g, time_g = results
            return np.array([S_Emp, S_TUR, S_hat_g, S_g])
        else:
            S_Emp, S_TUR, S_hat_g, S_g, time_tur, time_hat_g, time_g, theta_hat_g, theta_g = results
            data = np.load(file_name)
            return np.array([S_Emp, S_TUR, S_hat_g, S_g]), np.array(theta_hat_g), np.array(theta_g), data['J']
    print()
    print("=" * 70)
    print(f"  Starting EP estimation | System size: {N} | β = {beta:.4f}")
    print("=" * 70)

    # Total spin-flip attempts
    T = N * rep
    cap = None  # Cap on number of flips per spin (disabled)

    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
        print(f"No seed provided. Using random seed: {seed}")
        
    class DummyProgress:
        def __init__(self):
            self.value = 0

    progress = DummyProgress()
    print("[Sequential] Running on a single process")

    preload_depth = 5  # Number of spins to preload in parallel
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
    for i in tqdm(range(N//1), desc="Sequential spin progress"):
        if i in preload_threads:
            preload_threads[i].join()
            X_i, J_i, nflips = preload_results.pop(i)
            preload_threads.pop(i)
        else:
            X_i, J_i, nflips = get_spin_data(i, file_name, cap=cap)

        # Convert to torch tensors on appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#        device = "cpu"
        

        
        g_samples = torch.from_numpy(X_i).to(device).float() * 4 - 2  # {0,1} → {-1,1}
#        mask = np.ones(S_i.shape[1], dtype=bool)
#        mask[i] = False
#        # g = - 2*S_i[:, i] * S_i[:,mask], with S_ij = +1/-1
#        g = (S_i[:, i][:, None] ^ S_i)
#        g_samples = 2*(torch.from_numpy(g).to(device).float() * 2 - 1)  # {0,1} → {-1,1}
        J_i_t = torch.from_numpy(J_i).to(device)

#        mask = torch.ones(S_i.shape[1], dtype=bool)
#        mask[i] = False
##        
#        g_samples = -2*S_i_t[:, i][:, None] * S_i_t[:,mask]
        del X_i, J_i  # Free memory

        # Perform estimation
        calc_spin((i, N, beta, rep, T, file_name, temp_file_name_out, g_samples, J_i_t, nflips, seed))
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
        S_Emp, S_TUR, S_hat_g, S_g, time_tur, time_hat_g, time_g, theta_hat_g, theta_g = results
    else:
        S_Emp, S_TUR, S_hat_g, S_g, time_tur, time_hat_g, time_g = results

    print("\n[Results]")
    print(f"  Empirical     : {S_Emp:.6f}")
    print(f"  MTUR          : {S_TUR:.6f}   | Time: {time_tur:.2f} s")
    print(f"  Gaussian      : {S_hat_g:.6f}   | Time: {time_hat_g:.2f} s")
    print(f"  MaxEnt        : {S_g:.6f}   | Time: {time_g:.2f} s")
    print("-" * 70)

    if return_parameters:
        data = np.load(file_name)
        return np.array([S_Emp, S_TUR, S_hat_g, S_g]), theta_hat_g, theta_g, data["J"]
    else:
        return np.array([S_Emp, S_TUR, S_hat_g, S_g])
        
