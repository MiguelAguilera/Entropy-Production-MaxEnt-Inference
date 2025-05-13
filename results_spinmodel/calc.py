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

device = utils.set_default_torch_device()
torch.set_grad_enabled(False)


def calc_spin(beta, J, i, g_samples):
    verbose=False


    sigmas, times, thetas = {}, {}, {}

    data = ep_estimators.Dataset(g_samples=g_samples)

    # trn, tst = data.split_train_test()
    np.random.seed(123)
    trn, tst = data.split_train_test(holdout_fraction=0.2)


    stime = time.time()
    sigmas['Emp'] = spin_model.get_spin_empirical_EP(beta=beta, J=J, i=i, g_mean=data.g_mean)
    times['Emp'] = time.time() - stime

    if False:
        stime = time.time()
        tur_sol = ep_estimators.get_EP_MTUR(data=trn)
        sigmas['TUR'] = tur_sol.objective
        times['TUR'] = time.time() - stime

    # print( spin_model.get_spin_empirical_EP(beta=beta, J=J, i=i, g_mean=data.g_mean))
    # print( spin_model.get_spin_empirical_EP(beta=beta, J=J, i=i, g_mean=trn.g_mean))
    # print( spin_model.get_spin_empirical_EP(beta=beta, J=J, i=i, g_mean=tst.g_mean))
    # print()

# Emp=0.02383  TUR=0.00623  NR h a=0.02243  NR h a tst=0.02110  mem=4126.4mb:   2%|██                                                                                               | 21/1000 [00:37<29:15,  1.79s/it]^C^CTraceback (most recent call last):

# iteration 20, tst=0.020
    to_run = [
   #     ('N1'      ,      ep_estimators.get_EP_Newton, dict(data=trn, holdout_data=tst, max_iter=1) ),
  #      ('TUR'      ,      ep_estimators.get_EP_MTUR, dict(data=data) ),
#        ('NR'     ,      ep_estimators.get_EP_Newton, dict(data=trn, holdout_data=tst, trust_radius=1/4, adjust_radius=False)),

 #       ('NR h a'     ,      ep_estimators.get_EP_Newton, dict(data=trn, holdout_data=tst, trust_radius=1/4, 
                         # linsolve_eps=1e-2, tol=0,  adjust_radius=False, verbose=0) ),

        
#          ('G h'    ,      ep_estimators.get_EP_GradientAscent  , dict(data=trn, holdout_data=tst, tol=0, verbose=1,lr=.02) ),
#          ('G h'    ,      ep_estimators.get_EP_GradientAscent  , dict(data=trn, holdout_data=tst, tol=0) ),
         ('Gbb'    ,      ep_estimators.get_EP_GradientAscent  , dict(data=trn, holdout_data=tst, lr=1e-2, 
                                                                      use_BB=True, verbose=1, report_every=1, patience=20) ),
#          ('G'    ,      ep_estimators.get_EP_GradientAscent  , dict(data=data) ),
    ]
    utils.empty_torch_cache()

    stime = time.time()

    for k, f, kwargs in to_run:
        stime = time.time()
        res = f(**kwargs)
        # utils.torch_synchronize()
        times[k] = time.time() - stime
        if res.theta is not None:
            thetas[k] = utils.torch_to_numpy(res.theta)
        if res.trn_objective is None:  
            sigmas[k] = res.objective
        else: # holdout was used
            sigmas[k] = data.get_objective(res.theta)
            sigmas[k+' tst'] = res.objective
            #print(k, sigmas[k], res.objective, tst.get_objective(res.theta),trn.get_objective(res.theta))

        
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
            #asdf

    del data, trn, tst 

    return sigmas, times, thetas


def calc(file_name, max_spins=None):

    utils.empty_torch_cache()

    ep_sums   = defaultdict(float)
    time_sums = defaultdict(float)
    process = psutil.Process(os.getpid())

    print()
    print(f"[Loading] Reading data from file:\n  → {file_name}\n")
    data = np.load(file_name)
    S_bin  = data['S_bin'] # .astype('float32')*2-1 # torch.from_numpy(data["S_bin"]).to(device)*2-1
    rep, N = S_bin.shape
    F      = data['F'] # torch.from_numpy(data["F"]).to(device).bool()

    # if data['beta'] >= 3:
    #     return None

    if False:
        vvv=data['J'].reshape([1,-1])[0,:]
        plt.hist(vvv)
        #sns.kdeplot(vvv)
        plt.show()
        #asf

    J    = data['J']
    beta = data['beta']

    frequencies = F.sum(axis=0)/(N*rep)
    epdata = {'frequencies':frequencies, 'J': data['J'], 'beta': beta, 
                'thetas':{}, 'ep':{}, 'times':{}}

    if max_spins is None:
        spin_ids = np.arange(N)
    else:
        np.random.seed(123)
        spin_ids = np.random.choice(N, size=max_spins, replace=False)

    #spin_ids = [7,]
    pbar = tqdm(spin_ids, smoothing=0)

    print("=" * 70)
    print(f"  Starting EP estimation | System size: {N}")
    print("=" * 70)

    # Compute empirical EP
    stime = time.time()


    for i in pbar:
        if i % 40 == 0:
            utils.empty_torch_cache()

        g_samples = spin_model.get_g_observables_bin(S_bin, F, i)

        # g_samples = torch.concat([g_samples, torch.randn(1000, g_samples.shape[1])], dim=0)

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



