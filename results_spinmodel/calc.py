import sys, os, argparse, time, psutil
from collections import defaultdict 
from tqdm import tqdm
import numpy as np

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import torch

sys.path.insert(0, '..')
import utils
import ep_estimators
import spin_model

#device = utils.set_default_torch_device()
#torch.set_grad_enabled(False)


def calc_spin2(beta, J, i, g_samples):
    verbose=False
    import ep_multipartite

    sigmas, times, thetas = {}, {}, {}
    obj = ep_multipartite.EPEstimators(g_samples=g_samples) 

    stime = time.time()
    sigmas['Emp'] = spin_model.get_spin_empirical_EP(beta=beta, J=J, i=i, g_mean=obj.g_mean().cpu().numpy())
    times['Emp'] = time.time() - stime


    to_run = [ ('NR h a', obj.get_EP_Newton, dict(holdout=True, trust_radius=1/4, adjust_radius=True) ),]

    stime = time.time()

    for k, f, kwargs in to_run:
        stime = time.time()
        res = f(**kwargs)
        utils.torch_synchronize()
        sigmas[k] = res.objective

        times[k] = time.time() - stime
        if res.theta is not None:
            thetas[k] = res.theta.cpu().numpy()
            #sigmas[k] = data.get_objective(res.theta)
        if res.tst_objective is not None:
            sigmas[k+' tst'] = res.tst_objective

    del obj # , trn, tst 
    #utils.empty_torch_cache()

    return sigmas, times, thetas


def calc_spin(beta, J, i, g_samples):
    verbose=False


    sigmas, times, thetas = {}, {}, {}

    data = ep_estimators.Dataset(g_samples=g_samples)

    stime = time.time()
    sigmas['Emp'] = spin_model.get_spin_empirical_EP(beta=beta, J=J, i=i, g_mean=data.g_mean.cpu().numpy())
    times['Emp'] = time.time() - stime

    trn, tst = data.split_train_test()

    to_run = [
#        ('N1'      ,      ep_estimators.get_EP_Newton, dict(data=trn, holdout_data=tst, max_iter=1) ),
#        ('TUR'      ,      ep_estimators.get_EP_MTUR, dict(data=data) ),

        ('NR h a'     ,      ep_estimators.get_EP_Newton, dict(data=trn, holdout_data=tst, trust_radius=1/4, adjust_radius=True) ),

        
#          ('G h'    ,      obj.get_EP_GradientAscent  , dict(holdout=True) ),
#          ('G h'    ,      obj.get_EP_GradientAscent  , dict(holdout=True, max_iter=5000, lr=1e-2, tol=1e-8, use_Adam=False) ),
#          ('G'    ,      obj.get_EP_GradientAscent  , dict() ),
    ]
    #utils.empty_torch_cache()

    stime = time.time()

    for k, f, kwargs in to_run:
        stime = time.time()
        res = f(**kwargs)
        utils.torch_synchronize()
        times[k] = time.time() - stime
        if res.theta is not None:
            thetas[k] = res.theta.cpu().numpy()
            sigmas[k] = data.get_objective(res.theta)
        else:
            sigmas[k] = res
        if res.tst_objective is not None:
            sigmas[k+' tst'] = res.tst_objective

        
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

    del data, trn, tst 
    #utils.empty_torch_cache()

    return sigmas, times, thetas


def calc(file_name):
    ep_sums   = defaultdict(float)
    time_sums = defaultdict(float)
    process = psutil.Process(os.getpid())

    print()
    print(f"[Loading] Reading data from file:\n  → {file_name}\n")
    data = np.load(file_name)
    S      = data['S_bin']*2-1 # torch.from_numpy(data["S_bin"]).to(device)*2-1
    rep, N = S.shape
    F      = data['F'] # torch.from_numpy(data["F"]).to(device).bool()

    if False:
        vvv=data['J'].reshape([1,-1])[0,:]
        plt.hist(vvv)
        #sns.kdeplot(vvv)
        plt.show()
        #asf

    J    = data['J']
    beta = data['beta']

    frequencies = F.sum(axis=0)/(N*rep) # F.float().sum(axis=0).cpu().numpy()/(N*rep)
    epdata = {'frequencies':frequencies, 'J': data['J'], 'beta': beta, 
                'thetas':{}, 'ep':{}, 'times':{}}

    pbar = tqdm(range(N), smoothing=0)

    print("=" * 70)
    print(f"  Starting EP estimation | System size: {N}")
    print("=" * 70)

    # Compute empirical EP
    stime = time.time()

    for i in pbar:
        g_samples = spin_model.get_g_observables(S, F, i)
        
        res = calc_spin( beta, J, i, g_samples)

        epdata[i] = res 
        sigmas, times, thetas = res

        for k,v in sigmas.items():
            ep_sums[k]   += frequencies[i]*v
            time_sums[k] += times.get(k, np.nan)
            if k not in epdata['thetas']:
                epdata['thetas'][k] = []
            epdata['thetas'][k].append(thetas[k] if k in thetas else None)

        del g_samples, sigmas, times, thetas, res
        
        memory_usage = process.memory_info().rss / 1024 / 1024
        ll = [f'{k}={ep_sums[k]:3.5f} ' for k in ep_sums]
        pbar.set_description(" ".join(ll) + f' mem={memory_usage:.1f}mb')

    for k,v in ep_sums.items():
        epdata['ep'][k]=v
    for k,v in time_sums.items():
        epdata['times'][k]=v

    del S, F, J, data

    return epdata




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



