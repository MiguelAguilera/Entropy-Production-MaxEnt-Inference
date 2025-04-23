import os, argparse, time, pickle
import numpy as np
from collections import defaultdict 
from tqdm import tqdm

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"

import torch
import h5py
import hdf5plugin # This needs to be imported even thought its not explicitly used
from matplotlib import pyplot as plt
from methods_EP_multipartite import *
import gd 


LABELS = {'emp': 'Empirical', 'N1': 'Newton 1-step', 'TUR': 'MTUR', 'N2': 'Newton 2-step', 'GD': 'Grad Ascent'}

def set_default_device():
    """
    Determines the best available device for PyTorch operations and sets it as default.
    Returns:
        torch.device: The device that was set as default ('mps', 'cuda', or 'cpu')
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # Set MPS as default device
        torch.set_default_device(device)
        print(f"Set MPS as default device: {device}")
        import warnings
        warnings.filterwarnings("ignore", message="The operator 'aten::_linalg_solve_ex.result' is not currently supported on the MPS backend and will fall back to run on the CPU", category=UserWarning)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        # Set CUDA as default device
        torch.set_default_device(device)
        print(f"Set CUDA as default device: {device}")
    else:
        device = torch.device("cpu")
        # CPU is already the default, but we can set it explicitly
        torch.set_default_device(device)
        print("Set CPU as default device")
    return device


def calc_spin(S_i, J_i, i):
    verbose=False


    #print(i, time.time() - start_time_i)
    # S_i_t = torch.from_numpy(S_i).to(device)

    #if S_i.shape[1] <= 10:
    #    print(f"  [Warning] Skipping spin {i}: insufficient time steps")
    #    continue

    sigmas, times, thetas = {}, {}, {}

    start_time_i = time.time()
    sigmas['N1'], thetas['N1'], Da = get_EP_Newton(S_i, i)
    times['N1']  = time.time() - start_time_i

    start_time_i = time.time()
    sigmas['TUR'] = get_EP_MTUR(S_i, i)
    times['TUR']  = time.time() - start_time_i

    sigmas['emp']                            = exp_EP_spin_model(Da, J_i, i)
    if DO_NEWTON2:
        sigmas['N2'], theta2                 = get_EP_Newton2(S_i, theta1, Da, i)
        
    if GD_MODE > 0:
        start_time_gd_i = time.time()
        #x0=thetas['N1'] # 
        x0=torch.zeros_like(thetas['N1'])
        if GD_MODE == 2:
            sigmas['GD'], thetas['GD'] = get_EP_Adam2(S_i, theta_init=x0, i=i) 
        elif GD_MODE == 1:
            sigmas['GD'], thetas['GD'] = gd.get_EP_gd(S_i, i, x0=x0,  num_iters=1000)
        else:
            raise Exception('Uknown GD_MODE')
        times['GD'] = time.time() - start_time_gd_i
    else:
        times['GD'] = np.nan
        sigmas['GD'] = np.nan
        thetas['GD'] = None

    return sigmas, times, thetas


def calc(file_name):
    data = np.load(file_name)
    N, rep = data['S'].shape

    print()
    print("=" * 70)
    print(f"  Starting EP estimation | System size: {N} ")
    print("=" * 70)

    print(f"[Loading] Reading data from file:\n  → {file_name}\n")

    H = data['H']
    S = torch.from_numpy(data["S"].astype("float32") * 2 - 1).to(device)
    F = torch.from_numpy(data["F"]).to(device)
    assert(np.all(H==0))  # We do not support local fields in our analysis

    J = torch.from_numpy(data['J']).to(device)

    ep_sums   = defaultdict(float)
    time_sums = defaultdict(float)

    time_gd  = time_N1 = 0
    start_time = time.time()

    frequencies = F.float().sum(axis=1).cpu().numpy()/(N*rep)

    data = {'frequencies':frequencies}

    pbar = tqdm(range(N))

    for i in pbar:
        # S_i = S[:, F[i]]
        indices = torch.where(F[i])[0]  # Get indices where F[i] is True
        S_i = torch.index_select(S, 1, indices)

        res = calc_spin( S_i, J[i,:], i )
        data[i] = res 
        sigmas, times, thetas = res

        for k,v in sigmas.items():
            ep_sums[k]   += frequencies[i]*v
            time_sums[k] += times.get(k, np.nan)

        pbar.set_description(f'emp={ep_sums['emp']:1f}, N1={ep_sums['N1']:1f} GD={ep_sums['GD']:1f}')

    for k,v in ep_sums.items():
        data[k]=v

    print(f"\n[Results] {time.time()-start_time:3f}s")
    for k,lbl in LABELS.items():
        if k in ep_sums:
            print(f"  EP ({lbl:15s})    : {ep_sums[k]:.6f}   {time_sums[k]:3f}s")
    print("-" * 70)

    out_filename = os.path.dirname(file_name)+'/epdata_' + os.path.basename(file_name)+'.pkl'
    with open(out_filename, 'wb') as file:
        pickle.dump(data, file)
        print(f'Saved to {out_filename}')

    return data



device = set_default_device()
torch.set_grad_enabled(False)

DO_NEWTON2 = False
GD_MODE = 2  # 0 for no GD
             # 1 for pytorch Adam
             # 2 for our Adam

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Estimate EP for file.")
    parser.add_argument("file_name", type=str)

    args = parser.parse_args()

    calc(args.file_name)



