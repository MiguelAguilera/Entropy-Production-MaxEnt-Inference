import os, argparse, time, pickle, psutil
import numpy as np
from collections import defaultdict 
from tqdm import tqdm
import gc

import matplotlib.pyplot as plt 
import seaborn as sns
            
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"

import torch
from methods_EP_multipartite import *


LABELS = {'emp'  : 'Empirical', 
          'N1'   : 'Newton 1-step', 
          'TUR'  : 'MTUR', 
          'NS'   : 'Newton Steps', 
          'GD'   : 'Grad Ascent',
          'BFGS' : 'BFGS'}

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


def calc_spin(S_i, J_i, i, grad=False, newton=False):
    verbose=False

    #print(i, time.time() - start_time_i)
    # S_i_t = torch.from_numpy(S_i).to(device)

    #if S_i.shape[1] <= 10:
    #    print(f"  [Warning] Skipping spin {i}: insufficient time steps")
    #    continue

    sigmas, times, thetas = {}, {}, {}

    stime = time.time()
    sigmas['TUR'] = get_EP_MTUR(S_i, i)
    times[ 'TUR']  = time.time() - stime

    stime = time.time()
    sigmas['N1'], theta_N1, Da = get_EP_Newton(S_i, i)
    thetas['N1'] = theta_N1.cpu().numpy()
    times[ 'N1']  = time.time() - stime

    stime = time.time()
    sigmas['emp'] = exp_EP_spin_model(Da, J_i, i)
    times[ 'emp']  = time.time() - stime

    if newton:
        stime = time.time()
        sigmas['NS'], theta2  = get_EP_Newton_steps(S_i, theta_init=theta_N1, sig_init=sigmas['N1'], Da=Da, i=i)
        thetas['NS'] = theta2.cpu().numpy()
        times[ 'NS'] = time.time() - stime
        if sigmas['NS']>100:
            print('HUGE VALUE!', sigmas['NS'],get_objective(S_i, Da=Da, theta=theta2, i=i))
            #print(theta2)
            #print(Da)
            #print('tmax',torch.abs(theta2).max())
            #print('dmax',torch.abs(Da).max())
            #raise Exception()
    else:
        sigmas['NS'] = times['NS'] = np.nan
        thetas['NS'] = np.zeros(theta_N1.shape)

        
    if grad: #  > 0:
        x0=torch.zeros_like(theta_N1)
        stime = time.time()
        sigmas['GD'], ctheta  = get_EP_Adam(S_i, Da=Da, theta_init=x0, i=i) 
        thetas['GD'] = ctheta.cpu().numpy()
        times[ 'GD']  = time.time() - stime

        #sigmas['BFGS'], ctheta = get_EP_BFGS(S_i, Da=Da, theta_init=torch.zeros_like(theta_N1), i=i) 
        #thetas['BFGS']  = ctheta.cpu().numpy()
        #times['BFGS'] = time.time() - start_time_gd_i

        if False:  # histogram of g outcomes
            ctheta2 = torch.concatenate([ctheta[:i], torch.zeros(1), ctheta[i:]]) 
            stats = (ctheta2@S_i.T).cpu().numpy()
            stats -= stats.mean()
            print(np.mean( stats**3 ) )
            print(np.mean( stats**4 ) - 3*np.mean( stats**2 )**2  )
            #print(np.mean( (stats - stats.mean())**4-3*np.mean(stats - stats.mean())**2)
            sns.kdeplot( stats, label='Original')# , bins=20, color='red', alpha=0.3, label='Original') 
            
            stats2 = np.random.normal(loc=0, scale=stats.std(), size=len(stats))
            stats2 -= stats2.mean()
            #plt.hist( stats, bins=20, alpha=0.3, color='k', label='Gaussian')
            sns.kdeplot( stats2, label='Gaussian')
            #plt.yscale('log')

            plt.legend() 
            

            plt.show()
            #asdf

    else:
        sigmas['GD'] = times['GD'] = np.nan
        thetas['GD'] = np.zeros(theta_N1.shape)

    return sigmas, times, thetas


def calc(file_name, overwrite=False, newton=False, grad=False):
    ep_sums   = defaultdict(float)
    time_sums = defaultdict(float)
    start_time = time.time()
    process = psutil.Process(os.getpid())

    epdata = None

    out_filename = os.path.dirname(file_name)+'/epdata_' + os.path.basename(file_name)+'.pkl'
    if os.path.exists(out_filename):
        print(f'{out_filename} exists! '+('loading' if not overwrite else 'overwriting'))
        if not overwrite:
            with open(out_filename, 'rb') as file:
                epdata = pickle.load(file)
                

    if epdata is None:
        print()
        print(f"[Loading] Reading data from file:\n  → {file_name}\n")
        data = np.load(file_name)
        H = data['H']
        assert(np.all(H==0))  # We do not support local fields in our analysis
        with torch.no_grad():
            S      = torch.from_numpy(data["S"]).to(device)
            rep, N = S.shape
            F      = torch.from_numpy(data["F"]).to(device).bool()

            if False:
                vvv=data['J'].reshape([1,-1])[0,:]
                plt.hist(vvv)
                #sns.kdeplot(vvv)
                plt.show()
                #asf

            J = torch.from_numpy(data['J']).to(device)

            frequencies = F.float().sum(axis=0).cpu().numpy()/(N*rep)
            epdata = {'frequencies':frequencies, 'J': data['J'], 'beta': data['beta']}

            pbar = tqdm(range(N))

            print("=" * 70)
            print(f"  Starting EP estimation | System size: {N}")
            print("=" * 70)

            for i in pbar:
                S_i = S[F[:,i],:].to(torch.float32) * 2 - 1
                # S_i = torch.index_select(Sraw, 0, torch.where(F[:,i])[0] ).to(torch.float32) * 2 - 1

                res = calc_spin( S_i.contiguous(), J[i,:].contiguous(), i , grad=grad, newton=newton)

                epdata[i] = res 
                sigmas, times, thetas = res

                for k,v in sigmas.items():
                    ep_sums[k]   += frequencies[i]*v
                    time_sums[k] += times.get(k, np.nan)
                    kk = 'theta_'+k
                    if kk not in epdata:
                        epdata[kk] = []
                    epdata[kk].append(thetas.get(k, np.nan))

                del S_i, sigmas, times, thetas, res
                gc.collect()
                memory_info  = process.memory_info()
                memory_usage = memory_info.rss / 1024 / 1024
                pbar.set_description(f'emp={ep_sums['emp']:3.5f}, N1={ep_sums['N1']:3.5f} NS={ep_sums['NS']:3.5f} GD={ep_sums['GD']:3.5f} mem={memory_usage:.1f}mb')

            for k,v in ep_sums.items():
                epdata[k]=v


            with open(out_filename, 'wb') as file:
                pickle.dump(epdata, file)
                print(f'Saved to {out_filename}')

    print(f"\n[Results] {time.time()-start_time:3f}s")
    for k,lbl in LABELS.items():
        if k in epdata and not np.isnan(epdata[k]):
            print(f"  EP ({lbl:15s})    : {epdata[k]:.6f}   {time_sums.get(k,np.nan):3f}s")
    print("-" * 70)


    return epdata



device = set_default_device()
torch.set_grad_enabled(False)

# DO_NEWTON2 = True
# GD_MODE = 0  # 0 for no GD
#              # 1 for pytorch Adam
#              # 2 for our Adam

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Estimate EP for file.")
    parser.add_argument("file_name", type=str)

    args = parser.parse_args()

    calc(args.file_name)



