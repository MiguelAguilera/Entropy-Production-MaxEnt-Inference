import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import re
import h5py
import hdf5plugin # This needs to be imported even thought its not explicitly used
import time
import os
from joblib import Parallel, delayed
from methods_EP_multipartite import *
import gc
from threading import Thread

# -------------------------------
# Entropy Production Calculation Functions
# -------------------------------


#def get_spin_data(i, file_name):
#    with h5py.File(file_name, 'r') as f:
#        F_i = f['F'][i]             # loads only 1 row of F
#        S_i = f['S'][:, F_i].astype('float32') * 2 - 1  # convert back to {-1, 1}
#        J_i = f['J'][i, :]          # loads only 1 row of J
#        return S_i, J_i
def get_spin_data(i, file_name):
    data = np.load(file_name)

    F_i = data["F"][i]
    S_i = data["S"][:, F_i].astype("float32") * 2 - 1  # convert {0,1} → {-1,1}
    J_i = data["J"][i, :]

    return S_i, J_i


def select_device(threshold=0.5):
    if not torch.cuda.is_available():
        return torch.device("cpu")

    device = torch.device("cuda")
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory

    usage = max(allocated, reserved) / total
    if usage > threshold:
        print(f"⚠️ GPU usage {usage:.2%} exceeds threshold — switching to CPU.")
        return torch.device("cpu")

    return device
    
    
def calc_spin(i_args):
    i, N, T, file_name, file_name_out = i_args
    S_i, J_i = get_spin_data(i, file_name)


#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    S_i_t = torch.from_numpy(S_i).to(device)
#    J_i_t = torch.from_numpy(J_i).to(device)
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    device = select_device()
    S_i_t = torch.from_numpy(S_i).to(device)
    J_i_t = torch.from_numpy(J_i).to(device)

    if S_i.shape[1] <= 10:
        print(f"  [Warning] Skipping spin {i}: insufficient time steps")
        return None

    # Randomly assign device
    if torch.cuda.is_available() and torch.rand(1).item() < 1.:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
#    print(f"[Spin {i}] Running on device: {device}")
    S_i_t.to(device)
    J_i_t.to(device)
    Pi = S_i_t.shape[1] / T

    t0 = time.time()
    MTUR = Pi * get_EP_MTUR(S_i_t, i)
    time_tur = time.time() - t0

    t0 = time.time()
    sig_N1, theta_N1, Da = get_EP_Newton(S_i_t, i)
    N1 = Pi * sig_N1
    theta_N1_np = theta_N1.detach().cpu().numpy()
    time_n1 = time.time() - t0

    sig_N2 , theta_N2 = get_EP_Newton2(S_i_t, theta_N1, Da, i)   # Second newton step
##    sig_N2, theta_N2 = get_EP_Newton2(S_i_t, theta_N2.clone(), Da, i)   # Third newton step
    for r in range(2):
        sig_N2, theta_N2 = get_EP_Newton2(S_i_t, theta_N2.clone(), Da, i)   # Fourth newton step
#    sig_N2, theta_N2 = get_EP_Adam(S_i_t, theta_N1, Da, i)
#    sig_N2, theta_N2 = get_EP_BFGS(S_i_t, theta_N1, Da, i) 
#    sig_N2, theta_N2 = get_EP_gd(S_i_t, i, x0=theta_N1.clone().detach().requires_grad(True),  num_iters=1000)
    
    N2 = Pi * sig_N2
    theta_N2_np = theta_N2.detach().cpu().numpy()
    time_n2 = time.time() - t0
    

    Emp = Pi * exp_EP_spin_model(Da, J_i_t, i)
    
#    sig_Adam, theta_Adam = get_EP_Adam(S_i_t, theta_N1, Da, i)
#    time_adam = time.time() - t0

    # Modify output file name to include spin index
    spin_file_name = file_name_out.replace(".h5", f"_spin_{i:06d}.npz")

    # Save directly to .npz file
    np.savez(
        spin_file_name,
        MTUR=MTUR,
        time_tur=time_tur,
        N1=N1,
        theta_N1=theta_N1_np,
        time_n1=time_n1,
        N2=N2,
        theta_N2=theta_N2_np,
        time_n2=time_n2,
        Emp=Emp
    )
    
    torch.cuda.empty_cache()
    gc.collect()


        
def merge_spins(file_name_out, N):
    """
    Merge individual spin HDF5 files into a single merged file.

    Parameters:
        file_name_out (str): Base output file (e.g., 'data_N_100_beta_2.5.h5')
        N (int): System size (number of spins)

    Returns:
        str: Path to the merged HDF5 file
    """
    import os
    import h5py

    print(f"[Merging] Target merged file: {file_name_out}")
    
    with h5py.File(file_name_out, 'w') as f_out:
        for i in range(N):
            spin_file = file_name_out.replace(".h5", f"_spin_{i:06d}.npz")
            if not os.path.exists(spin_file):
                print(f"[Error] Spin file not found: {spin_file}")
                exit()

            data = np.load(spin_file)
            for key in data:
                dataset_name = f"{key}_{i}"
                f_out.create_dataset(dataset_name, data=data[key])
    for i in range(N):
        spin_file = file_name_out.replace(".h5", f"_spin_{i:06d}.npz")
        os.remove(spin_file)
    return file_name_out
    
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
        

def calc_spin_group(group_args):
    indices, N, T, file_name, file_name_out, progress, lock = group_args
    for i in indices:
        calc_spin((i, N, T, file_name, file_name_out))
        torch.cuda.empty_cache()
        gc.collect()
        with lock:
            progress.value += 1
    
def calc(N, rep, file_name, file_name_out, return_parameters=False, num_processes = 2):
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

#    with h5py.File(file_name, 'r') as f:
#        J = f['J'][:]
#        H = f['H'][:]
#        assert(np.all(H==0))  # We do not support local fields in our analysis

    data = np.load(file_name)
    J = data["J"]
    H = data["H"]
    assert np.all(H == 0), "Non-zero local fields are not supported"

    # Initialize accumulators
    S_Emp = S_TUR = S_N1 = S_N2 = S_Adam = 0
    time_emp = time_tur = time_n1 = time_n2 = time_adam = 0
    T = N * rep  # Total spin-flip attempts

#    num_processes = min(mp.cpu_count(),6)
#    group_size = int(np.ceil(N / num_processes))
#    grouped_indices = [list(range(i, min(i + group_size, N))) for i in range(0, N, group_size)]
#    args_list = [(indices, N, T, file_name, file_name_out) for indices in grouped_indices]

#    Parallel(
#        n_jobs=num_processes,
#        backend="multiprocessing",
#        prefer="processes"
#    )(delayed(calc_spin_group)(args) for args in args_list)

    group_size = int(np.ceil(N / num_processes))
    grouped_indices = [list(range(i, min(i + group_size, N))) for i in range(0, N, group_size)]

    print(f"[Parallel] Using {num_processes} processes via torch.multiprocessing")

    ctx = mp.get_context("spawn")  # Required for CUDA
    manager = mp.Manager()
    progress = manager.Value('i', 0)
    lock = manager.Lock()

    total_tasks = sum(len(g) for g in grouped_indices)

    # tqdm wrapper around pool.map
    with tqdm(total=total_tasks, desc="Global spin progress", position=0) as pbar:
        def update_bar():
            prev = 0
            while progress.value < total_tasks:
                with lock:
                    new = progress.value
                if new > prev:
                    pbar.update(new - prev)
                    prev = new
                time.sleep(0.1)

        updater = Thread(target=update_bar)
        updater.start()

        args_list = [(indices, N, T, file_name, file_name_out, progress, lock) for indices in grouped_indices]
        with ctx.Pool(processes=num_processes) as pool:
            pool.map(calc_spin_group, args_list)

        updater.join()

#    ctx = mp.get_context("spawn")  # "spawn" is required for CUDA
#    with ctx.Pool(processes=num_processes) as pool:
#        pool.map(calc_spin_group, args_list)
        
        
    # Merge individual spin HDF5 files into a single merged file.
    merge_spins(file_name_out, N)
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
    parser.add_argument("--merge", action="store_true", help="If set, merge all spin outputs for the given beta")

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

    if args.merge:
        print("=" * 70)
        print(f"[MERGE MODE] Merging spin results for beta = {beta}")
        print(f"Output merged file: {file_out}")
        print("=" * 70)
        merge_spins(file_out, args.N)
    else:
        print("=" * 70)
        print(f"Running single spin calculation for i = {args.i}, beta = {beta:.4f}")
        print(f"Input:  {file_in}")
        print(f"Output: {file_out}")
        print("=" * 70)
        calc_spin((args.i, args.N, T, file_in, file_out))


