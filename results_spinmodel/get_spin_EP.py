import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import re
import h5py
import hdf5plugin # This needs to be imported even thought its not explicitly used
import time
import os, sys
import gc
from threading import Thread
import utils


sys.path.insert(0, '..')
from ep_multipartite import EPEstimators

# -------------------------------
# Entropy Production Calculation Functions
# -------------------------------


#def get_spin_data(i, file_name):
#    with h5py.File(file_name, 'r') as f:
#        F_i = f['F'][i]             # loads only 1 row of F
#        S_i = f['S'][:, F_i].astype('float32') * 2 - 1  # convert back to {-1, 1}
#        J_i = f['J'][i, :]          # loads only 1 row of J
#        return S_i, J_i
def get_spin_data(i, file_name, cap=None):
    data = np.load(file_name)

    F_i = data["F"][:, i]
    S_i = data["S_bin"][F_i, :]
    J_i = data["J"][i, :]
    nflips = S_i.shape[0]
    if cap is None:
        return S_i, J_i, nflips
    else:
        return S_i[:cap, :], J_i, nflips


#def select_device(threshold=0.6):
#    if not torch.cuda.is_available():
#        return torch.device("cpu")

#    device = torch.device("cuda")
#    allocated = torch.cuda.memory_allocated(device)
#    reserved = torch.cuda.memory_reserved(device)
#    total = torch.cuda.get_device_properties(device).total_memory

#    usage = max(allocated, reserved) / total
#    if usage > threshold:
#        print(f"⚠️ GPU usage {usage:.2%} exceeds threshold — switching to CPU.")
#        return torch.device("cpu")

#    return device
    
#def calc_spin(i_args):
#    i, N, beta, rep, T, file_name, file_name_out = i_args

#    cap=1000000
#    S_i, J_i, nflips = get_spin_data(i, file_name, cap= cap)
    
def calc_spin(i_args):
    i, N, beta, rep, T, file_name, file_name_out, S_i_t, J_i_t, nflips = i_args


    num_chunks = 5
    if S_i_t.shape[0] <= 100:
        print(f"  [Warning] Skipping spin {i}: insufficient spin flips")
        return None
#    print( S_i_t.shape[1])
#    print(f"[Spin {i}] Running on device: {device}")
#    S_i_t.to(device)
#    J_i_t.to(device)
    Pi = nflips / T

    t0 = time.time()
    time_tur = time.time() - t0


    t0 = time.time()
    
    ep_estimator = EPEstimators(S_i_t, i, num_chunks=5)
    
    sig_MTUR, theta_N1, _ = ep_estimator.get_EP_MTUR()
    MTUR = Pi * sig_MTUR
    sig_N1, theta_N1, _ = ep_estimator.get_EP_Newton(max_iter=1, holdout=True, adjust_radius=True)
    
    N1 = Pi * sig_N1
    theta_N1_np = theta_N1.detach().cpu().numpy()
    time_n1 = time.time() - t0


    Emp = Pi * beta * float(utils.remove_i(J_i_t,i) @ ep_estimator.g_mean())
    del J_i_t
    torch.cuda.empty_cache()
    

    sig_N2, theta_N2, _ = ep_estimator.get_EP_Newton(trust_radius=0.25, holdout=True, adjust_radius=False)
    
    N2 = Pi * sig_N2
    theta_N2_np = theta_N2.detach().cpu().numpy()
    time_n2 = time.time() - t0
    
    
    del S_i_t
    torch.cuda.empty_cache()
    gc.collect()

    # Save into h5
    with h5py.File(file_name_out, 'a') as f_out:
        spin_group_name = f"spins/{i}"
        if spin_group_name in f_out:
            del f_out[spin_group_name]  # 🚀 delete old group if it already exists
        group = f_out.create_group(spin_group_name)
        group.create_dataset("MTUR", data=MTUR)
        group.create_dataset("time_tur", data=time_tur)
        group.create_dataset("N1", data=N1)
        group.create_dataset("theta_N1", data=theta_N1_np)
        group.create_dataset("time_n1", data=time_n1)
        group.create_dataset("N2", data=N2)
        group.create_dataset("theta_N2", data=theta_N2_np)
        group.create_dataset("time_n2", data=time_n2)
        group.create_dataset("Emp", data=Emp)
        
        
def load_results_from_file(file_name_out, N, return_parameters=False):
    S_Emp = S_TUR = S_N1 = S_N2 = time_tur = time_n1 = time_n2 = 0
    theta_N1_list = []
    theta_N2_list = []

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

    if not return_parameters:
        return S_Emp, S_TUR, S_N1, S_N2, time_tur, time_n1, time_n2
    else:
        return (
            S_Emp, S_TUR, S_N1, S_N2,
            time_tur, time_n1, time_n2,
            np.array(theta_N1_list), np.array(theta_N2_list)
        )
        

##def calc_spin_group(group_args):
##    indices, N, beta, rep, T, file_name, file_name_out, progress, lock, check_memory = group_args

##    cuda_available = torch.cuda.is_available()
##    if cuda_available and check_memory:
##        time.sleep(indices[0]/N*4)
##    device = torch.device("cuda") if cuda_available else torch.device("cpu")

##    cap=1000000
##    S_i, J_i, nflips = get_spin_data(i, file_name, cap= cap)
##    
##    for i in indices:
##        if i + 1 < N:
##            preload_thread = Thread(target=lambda: get_spin_data(i+1, file_name, cap= cap))
##            preload_thread.start()
##        calc_spin((i, N, beta, rep, T, file_name, file_name_out, S_i, J_i, nflips))
##        with lock:
##            progress.value += 1
##        del S_i, J_i
##        # Wait for next preload to finish
##        if i + 1 < N:
##            preload_thread.join()
##            S_i, J_i, nflips = get_spin_data(i+1, file_name, cap= cap)
     
def calc(N, beta, rep, file_name, file_name_out, return_parameters=False, parallel=False, num_processes = 2, overwrite=True, check_memory=True):
    """
    Compute entropy production rate (EP) estimates using multiple methods for a spin system.

    Parameters:
        N (int): System size.
        rep (int): Number of repetitions.

    Returns:
        np.ndarray: EP estimates [empirical, MTUR, Newton-1, Newton-2]
    """
    data = np.load(file_name)
    J = data["J"]
    
    if os.path.exists(file_name_out) and not overwrite:
        print(f"[Info] Output file '{file_name_out}' already exists. Skipping computation.")
        results = load_results_from_file(file_name_out, N, return_parameters)
        if not return_parameters:
            S_Emp, S_TUR, S_N1, S_N2, time_tur, time_n1, time_n2 = results
            return np.array([S_Emp, S_TUR, S_N1, S_N2])
        else:
            S_Emp, S_TUR, S_N1, S_N2, time_tur, time_n1, time_n2, theta_N1, theta_N2 = results
            return np.array([S_Emp, S_TUR, S_N1, S_N2]), np.array(theta_N1), np.array(theta_N2), J
#    N//=50
    beta_str = re.search(r'_beta_([0-9.]+)', file_name).group(1)
    beta = float(beta_str)
    print()
    print("=" * 70)
    print(f"  Starting EP estimation | System size: {N} | β = {beta:.4f}")
    print("=" * 70)



    T = N * rep  # Total spin-flip attempts
#    cap= int(rep*0.3)
    cap=None
    if not parallel:
    
        class DummyProgress:
            def __init__(self):
                self.value = 0

        progress = DummyProgress()
        print(f"[Sequential] Running on a single process")

        preload_depth = 4

        preload_threads = {}
        preload_results = {}

        # Preload first 4 spins
        for preload_i in range(min(preload_depth, N)):
            def preload_func(j=preload_i):
                preload_results[j] = get_spin_data(j, file_name, cap=cap)
            thread = Thread(target=preload_func)
            thread.start()
            preload_threads[preload_i] = thread

        temp_file_name_out = file_name_out + ".tmp"
        # If temp file exists from a crash, remove it
        if os.path.exists(temp_file_name_out):
            os.remove(temp_file_name_out)

        # Now start computing
        for i in tqdm(range(N), desc="Sequential spin progress"):

            # Wait for spin i to be ready
            if i in preload_threads:
                preload_threads[i].join()
                S_i, J_i, nflips = preload_results.pop(i)
                preload_threads.pop(i)
            else:
                S_i, J_i, nflips = get_spin_data(i, file_name, cap=cap)
    
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#            if torch.cuda.is_available():
#                S_i_tensor = torch.as_tensor(S_i).pin_memory()
#                S_i_t = S_i_tensor.to(device, non_blocking=True).float() * 2 - 1
#                J_i_tensor = torch.as_tensor(J_i).pin_memory()
#                J_i_t = J_i_tensor.to(device, non_blocking=True)
#            else:
#            with torch.no_grad():
#            if True:
            S_i_t = (torch.from_numpy(S_i)).to(device).float()* 2 - 1  # convert {0,1} → {-1,1}
            J_i_t = torch.from_numpy(J_i).to(device)
            del S_i, J_i # Free RAM
            
            # Compute
            calc_spin((i, N, beta, rep, T, file_name, temp_file_name_out, S_i_t, J_i_t, nflips))
            progress.value += 1

            # Preload next spin if needed
            next_spin = i + preload_depth
            if next_spin < N and next_spin not in preload_threads:
                def preload_func(j=next_spin):
                    preload_results[j] = get_spin_data(j, file_name, cap=cap)
                thread = Thread(target=preload_func)
                thread.start()
                preload_threads[next_spin] = thread


        # Move data to file
        os.rename(temp_file_name_out, file_name_out)
        
        
#    else:
#        grouped_indices = np.array_split(np.arange(N), num_processes)
#        grouped_indices = [list(g) for g in grouped_indices]

#        print(f"[Parallel] Using {num_processes} processes via torch.multiprocessing")

#        ctx = mp.get_context("spawn")  # Required for CUDA
#        manager = ctx.Manager()
#        progress = manager.Value('i', 0)
#        lock = manager.Lock()
#        gpu_lock = manager.Lock()
#        
#        total_tasks = sum(len(g) for g in grouped_indices)

#        with tqdm(total=total_tasks, desc="Global spin progress", position=0) as pbar:
#            def update_bar():
#                prev = 0
#                while progress.value < total_tasks:
#                    with lock:
#                        new = progress.value
#                    if new > prev:
#                        pbar.update(new - prev)
#                        prev = new
#                    time.sleep(0.1)

#            updater = Thread(target=update_bar)
#            updater.start()

#            args_list = [(indices, N, beta, rep, T, file_name, file_name_out, progress, lock, gpu_lock) for indices in grouped_indices]
#            with ctx.Pool(processes=num_processes) as pool:
#                pool.map(calc_spin_group, args_list)

#            updater.join()

        
#    # Merge individual spin files into a single merged HDF5 file.
#    merge_spins(file_name_out, N)

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
#        theta_N1 = np.array(theta_N1_list)
#        theta_N2 = np.array(theta_N2_list)
        return np.array([S_Emp, S_TUR, S_N1, S_N2]), theta_N1, theta_N2, J
        
        
# -------------------------------
# Main script for individual spin EP calculation (for cluster computation)
# -------------------------------

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Compute EP for one spin at a given beta, or merge all spins for that beta.")
    parser.add_argument("i", type=int, help="Spin index i (0-based). Ignored if --merge is used.")
    parser.add_argument("beta", type=float, help="Beta value")
    parser.add_argument("N", type=int, help="System size")
    parser.add_argument("rep", type=int, help="Number of repetitions")
    parser.add_argument("--base_dir", type=str, default="~/MaxEntData",
                        help="Base directory where input files are stored (default: ~/MaxEntData)")
    parser.add_argument("--out_dir", type=str, default="ep_data/spin",
                        help="Directory to store output data (default: ep_data/spin)")
    parser.add_argument("--J0", type=float, default=1.0, help="Mean coupling J0 (default: 1.0)")
    parser.add_argument("--DJ", type=float, default=0.5, help="Variance DJ (default: 0.5)")
    parser.add_argument("--num_steps", type=int, default=128, help="Number of time steps (default: 128)")
    parser.add_argument("--patterns", type=int, default=None,
                        help="Hopfield pattern density (optional)")
#    parser.add_argument("--merge", action="store_true", help="If set, merge all spin outputs for the given beta")

    args = parser.parse_args()
    beta = round(args.beta, 8)
    T = args.N * args.rep

    # Expand user paths
    base_dir = os.path.expanduser(args.base_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Construct file paths
    if args.patterns is None:
        file_in = f"{base_dir}/sequential/run_reps_{args.rep}_steps_{args.num_steps}_{args.N:06d}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}.npz"
        file_out = f"{out_dir}/results_N_{args.N}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}.h5"
    else:
        file_in = f"{base_dir}/sequential/run_reps_{args.rep}_steps_{args.num_steps}_{args.N:06d}_beta_{beta}_patterns_{args.patterns}.npz"
        file_out = f"{out_dir}/results_N_{args.N}_beta_{beta}_patterns_{args.patterns}.h5"

#    if args.merge:
#        print("=" * 70)
#        print(f"[MERGE MODE] Merging spin results for beta = {beta}")
#        print(f"Output merged file: {file_out}")
#        print("=" * 70)
#        merge_spins(file_out, args.N)
#    else:
    print("=" * 70)
    print(f"Running single spin calculation for i = {args.i}, beta = {beta:.4f}")
    print(f"Input:  {file_in}")
    print(f"Output: {file_out}")
    print("=" * 70)
    calc_spin((args.i, args.N, T, file_in, file_out))


