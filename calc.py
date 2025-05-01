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


    to_run = [
        #('N1'     ,      obj.get_EP_Newton , dict()),
        #('N1v'       ,      obj.get_EP_Newton_steps, dict(max_iter=1, holdout=False,verbose=True) ),
        ('N1'      ,      obj.get_EP_Newton, dict(max_iter=1, holdout=True) ),
        ('TUR'      ,      obj.get_EP_MTUR        , dict(holdout=True)),
        ('NR h'     ,      obj.get_EP_Newton, dict(trust_radius=1/4, holdout=True) ),
        ('NR h a'     ,      obj.get_EP_Newton, dict(trust_radius=1/4, holdout=True, adjust_radius=True, verbose=True) ),

#         ('NR h'     ,      obj.get_EP_Newton, dict(holdout=True, trust_radius=1/4, solve_constrained=False, verbose=True) ),
#         ('NR na h'  ,      obj.get_EP_Newton, dict(holdout=True, trust_radius=1/4, solve_constrained=False, adjust_radius=True,verbose=True) ),
        
#         ('N h'      ,      obj.get_EP_Newton, dict(holdout=True,verbose=True) ),
#         ('N'        ,      obj.get_EP_Newton, dict(holdout=False,verbose=True) ),
#         ('NT h'     ,      obj.get_EP_Newton, dict(holdout=True, trust_radius=1/4, solve_constrained=True,verbose=True) ),
#         ('NT na h'  ,      obj.get_EP_Newton, dict(holdout=True, trust_radius=1/4, solve_constrained=True,adjust_radius=True,verbose=True) ),
        
# #        ('T h'    ,      obj.get_EP_TRON        , dict(holdout=True, trust_radius_init=1/4) ),
# #        ('T na h' ,      obj.get_EP_TRON        , dict(holdout=True, trust_radius_init=1/4, adjust_radius=False) ),
#         #('N h'    ,      obj.get_EP_Newton_steps, dict(holdout=True, trust_radius=1/4) ),
        
#          ('G h'    ,      obj.get_EP_GradAscent  , dict(holdout=True) ),
#          ('G'    ,      obj.get_EP_GradAscent  , dict() ),
    ]


    # Compute empirical EP for spin i
    sigmas['Emp'] = float(utils.remove_i(J_i,i) @ obj.g_mean())

    for k, f, kwargs in to_run:
        stime = time.time()
        res = f(**kwargs)
        times[k] = time.time() - stime
        sigmas[k] = res.sigma
        if res.theta is not None:
            thetas[k] = res.theta.cpu().numpy()
        if res.tst_sigma is not None:
            sigmas[k+' tst'] = res.tst_sigma

        
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
            ll = [f'{k}={ep_sums[k]:3.5f} ' for k in ep_sums]
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



