import sys, os, argparse, time, psutil
from collections import defaultdict 
from tqdm import tqdm
import numpy as np

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import torch

sys.path.insert(0, '..')
import utils
import ep_estimators
import observables
import spin_model

device = utils.set_default_torch_device()
torch.set_grad_enabled(False)


def calc_spin(beta, J, i, g_samples):
    data = observables.Dataset(g_samples=g_samples)
    np.random.seed(123)
    trn, val, tst = data.split_train_val_test(val_fraction=0.2, test_fraction=0.2)

    sigmas, times, thetas = {}, {}, {}

    stime = time.time()
    if data.nsamples == 0:
        sigmas['Emp'] = 0
    else:
        sigmas['Emp'] = spin_model.get_spin_empirical_EP(beta=beta, J=J, i=i, g_mean=data.g_mean)
    times['Emp'] = time.time() - stime


    to_run = [
        ('N1'      ,      ep_estimators.get_EP_Newton1Step, dict(data=trn, validation=val, test=tst) ),
        ('TUR'     ,      ep_estimators.get_EP_MTUR, dict(data=data) ),
#        ('NR'     ,      ep_estimators.get_EP_Newton, dict(data=trn, holdout_data=tst, trust_radius=1/4, adjust_radius=False)),

 #       ('NR h a'     ,      ep_estimators.get_EP_Newton, dict(data=trn, holdout_data=tst, trust_radius=1/4, 
                         # linsolve_eps=1e-2, tol=0,  adjust_radius=False, verbose=0) ),

        
#          ('G h'    ,      ep_estimators.get_EP_GradientAscent  , dict(data=trn, holdout_data=tst, tol=0, verbose=1,lr=.02) ),
#          ('G h'    ,      ep_estimators.get_EP_GradientAscent  , dict(data=trn, holdout_data=tst, tol=0) ),
         ('Gbb'    ,      ep_estimators.get_EP_Estimate  , dict(data=trn, validation=val, test=tst) ),
#          ('G'    ,      ep_estimators.get_EP_GradientAscent  , dict(data=data) ),
    ]
    utils.empty_torch_cache()

    stime = time.time()

    for k, f, kwargs in to_run:
        stime = time.time()
        sigmas[k], thetas[k] = f(**kwargs)
        # utils.torch_synchronize()
        times[k] = time.time() - stime
        # if res.theta is not None:
        #     thetas[k] = utils.torch_to_numpy(res.theta)
        # if res.trn_objective is None:  
        #     sigmas[k] = res.objective
        # else: # holdout was used
        #     sigmas[k] = res.trn_objective
        #     sigmas[k+' tst'] = res.objective
        #     #print(k, sigmas[k], res.objective, tst.get_objective(res.theta),trn.get_objective(res.theta))

        
    if False:  # histogram of g outcomes
        import matplotlib.pyplot as plt 
        import seaborn as sns
        ctheta2 = torch.concatenate([ctheta[:i], torch.zeros(1), ctheta[i:]]) 
        stats = utils.torch_to_numpy(ctheta2@S_i.T)
        stats -= stats.mean()
        sns.kdeplot( stats, label='Original')# , bins=20, color='red', alpha=0.3, label='Original') 
        
        stats2 = np.random.normal(loc=0, scale=stats.std(), size=len(stats))
        stats2 -= stats2.mean()
        sns.kdeplot( stats2, label='Gaussian')
        #plt.yscale('log')
        plt.legend() 
        plt.show()

    del data, trn, tst 

    return sigmas, times, thetas


def calc(file_name, max_spins=None):

    utils.empty_torch_cache()

    ep_sums   = defaultdict(float)
    time_sums = defaultdict(float)
    process = psutil.Process(os.getpid())

    print()
    print(f"[Loading] Reading data from file:\n  → {file_name}\n")
    data   = np.load(file_name)
    S_bin  = data['S_bin'] # .astype('float32')*2-1 # torch.from_numpy(data["S_bin"]).to(device)*2-1
    rep, N = S_bin.shape
    F      = data['F'] # torch.from_numpy(data["F"]).to(device).bool()

    J    = data['J']
    beta = data['beta']

    frequencies = F.sum(axis=0)/(N*rep)
    epdata = {'frequencies':frequencies, 'J': data['J'], 'beta': beta, 
                'thetas':{}, 'ep':{}, 'times':{}}

    if max_spins is None or max_spins >= N:
        spin_ids = np.arange(N)
    else:
        np.random.seed(123)
        spin_ids = np.random.choice(N, size=max_spins, replace=False)

    pbar = tqdm(spin_ids, smoothing=0)

    print("=" * 70)
    print(f"  Starting EP estimation | System size: {N}")
    print("=" * 70)

    # Compute empirical EP
    stime = time.time()


    for i in pbar:
        if i % 40 == 0:
            utils.empty_torch_cache()

        g_samples = observables.get_g_observables_bin(S_bin, F, i)
        if i == 5:
            g_samples = g_samples[:0,:]

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
        def perc(k):
            if k == 'Emp':
                return ''
            elif not utils.is_infnan(ep_sums[k]/ep_sums['Emp']):
                return f'({int(100*ep_sums[k]/ep_sums['Emp']):3d}%) ' 
            else:
                return '(---%) '
            
        ll = [f'{k}={ep_sums[k]:3.5f} ' + perc(k) for k in ep_sums]
        pbar.set_description("  ".join(ll) + f' mem={memory_usage:.1f}mb')

    #utils.empty_torch_cache()
    print(f'{time.time() - stime:3f}s')

    for k,v in ep_sums.items():
        epdata['ep'][k]=v
    for k,v in time_sums.items():
        epdata['times'][k]=v

    del S_bin, F, J, data

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



