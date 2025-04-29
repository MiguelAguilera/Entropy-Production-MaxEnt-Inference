import os, argparse, time, psutil
from collections import defaultdict 
from tqdm import tqdm

import numpy as np

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import torch
import utils
import ep_multipartite as epm
            


def calc_spin(S_i, J_i, i):
    verbose=False

    sigmas, times, thetas = {}, {}, {}

    obj = epm.EPEstimators(S_i, i)


    stime = time.time()
    sigmas['N1'], thetas['N1'] = obj.get_EP_Newton()
    times[ 'N1']  = time.time() - stime

    stime = time.time()
    sigmas['TUR'] = obj.get_EP_MTUR()
    times[ 'TUR'] = time.time() - stime

    stime = time.time()
    # Compute empirical EP for spin i
    sigmas['Emp'] = float(utils.remove_i(J_i,i) @ obj.g_mean())
    times[ 'Emp'] = time.time() - stime

    if False:
        stime = time.time()
        sigmas['Ntrst'], thetas['Ntrst'] = obj.get_EP_Newton_steps(newton_step_args=dict(delta=1))
        times[ 'Ntrst'] = time.time() - stime
    
    if False:
        stime = time.time()
        sigmas['Nthr'], thetas['Nthr'] = obj.get_EP_Newton_steps(newton_step_args=dict(th=0.01))
        times[ 'Nthr'] = time.time() - stime

    if False:
        stime = time.time()
        sigmas['Ntron'], thetas['Ntron'] = obj.get_EP_TRON()
        times[ 'Ntron'] = time.time() - stime


    if True:
        stime = time.time()
        sigmas['Nhld'], thetas['Nhld'] = obj.get_EP_Newton_steps_holdout()
        times[ 'Nhld'] = time.time() - stime

    if False:
        stime = time.time()
        sigmas['Nhld2'], thetas['Nhld2']  = obj.get_EP_Newton_steps_holdout(
            newton_step_args=dict(delta=.1))
        times[ 'Nhld2'] = time.time() - stime
        
    if True: # Grad
        x0=torch.zeros(len(J_i)-1)
        stime = time.time()
        sigmas['Grad'], thetas['Grad'] = obj.get_EP_Adam(theta_init=x0) 
        times[ 'Grad']  = time.time() - stime

        #sigmas['BFGS'], ctheta = get_EP_BFGS(S_i, g=g, theta_init=torch.zeros_like(theta_N1), i=i) 
        #thetas['BFGS']  = ctheta.cpu().numpy()
        #times['BFGS'] = time.time() - start_time_gd_i

        if False:  # histogram of g outcomes
            import matplotlib.pyplot as plt 
            import seaborn as sns
            ctheta2 = torch.concatenate([ctheta[:i], torch.zeros(1), ctheta[i:]]) 
            stats = (ctheta2@S_i.T).cpu().numpy()
            stats -= stats.mean()
            sns.kdeplot( stats, label='Original')# , bins=20, color='red', alpha=0.3, label='Original') 
            
            stats2 = np.random.normal(loc=0, scale=stats.std(), size=len(stats))
            stats2 -= stats2.mean()
            sns.kdeplot( stats2, label='Gaussian')
            #plt.yscale('log')

            plt.legend() 
            

            plt.show()
            #asdf

    del obj
    utils.empty_cache()

    for k in sigmas:
        sigmas[k] = float(sigmas[k])
        if k in thetas:
            thetas[k] = thetas[k].cpu().numpy()

    return sigmas, times, thetas


def calc(file_name):
    ep_sums   = defaultdict(float)
    time_sums = defaultdict(float)
    process = psutil.Process(os.getpid())

    print()
    print(f"[Loading] Reading data from file:\n  → {file_name}\n")
    data = np.load(file_name)
    assert(np.all(data['H']==0))  # We do not support local fields in our analysis
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
        epdata = {'frequencies':frequencies, 'J': data['J'], 'beta': data['beta'], 
                  'thetas':{}, 'ep':{}, 'times':{}}

        pbar = tqdm(range(N))

        print("=" * 70)
        print(f"  Starting EP estimation | System size: {N}")
        print("=" * 70)

        for i in pbar:
            S_i = S[F[:,i],:].to(torch.float32) * 2 - 1

            res = calc_spin( S_i.contiguous(), J[i,:].contiguous(), i)

            epdata[i] = res 
            sigmas, times, thetas = res

            for k,v in sigmas.items():
                ep_sums[k]   += frequencies[i]*v
                time_sums[k] += times.get(k, np.nan)
                if k not in epdata['thetas']:
                    epdata['thetas'][k] = []
                epdata['thetas'][k].append(thetas[k] if k in thetas else None)

            del S_i, sigmas, times, thetas, res
            
            memory_usage = process.memory_info().rss / 1024 / 1024
            ll = [f'{k}={ep_sums[k]:3.5f} ' for k in ['Emp', 'N1', 'Ntrst','Nthr','Nhld','Nhld2','Grad'] if k in ep_sums]
            pbar.set_description(" ".join(ll) + f' mem={memory_usage:.1f}mb')

        for k,v in ep_sums.items():
            epdata['ep'][k]=v
        for k,v in time_sums.items():
            epdata['times'][k]=v

        del S, F, J
    del data

    return epdata



device = utils.set_default_device()
torch.set_grad_enabled(False)

# DO_NEWTON2 = True
# GD_MODE = 0  # 0 for no GD
#              # 1 for pytorch Adam
#              # 2 for our Adam

if __name__ == "__main__":
    raise Exception()

    parser = argparse.ArgumentParser(description="Estimate EP for file.")
    parser.add_argument("file_name", type=str)
    parser.add_argument("--overwrite", action="store_true",  default=False, help="Overwrite existing files.")

    args = parser.parse_args()

    calc(args.file_name, overwrite=args.overwrite)



