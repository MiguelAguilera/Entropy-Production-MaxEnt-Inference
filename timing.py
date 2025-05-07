import time, os
import numpy as np
from tqdm import tqdm

#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable torch fallback for MPS backend
import torch

import spin_model
import utils

test_old=False
if test_old:
    print('testing old version')
    import ep_multipartite as epm
else:
    print('testing new version')
    import ep_estimators as epm
# The following allows torch to use GPU for computation
# utils.set_default_torch_device()                
print(torch.get_default_device())
N    = 10   # system size
k    = 6    # avg number of neighbors in sparse coupling matrix
beta = 2.0   # inverse temperature
rep =100000
N=1000
rep=10000
np.random.seed(42) # Set seed for reproducibility

stime = time.time()
J    = spin_model.get_couplings_random(N=N, k=k)
S, F = spin_model.run_simulation(beta=beta, J=J, samples_per_spin=rep)
num_samples_per_spin, N = S.shape
total_flips = N * num_samples_per_spin  # Total spin-flip attempts
print(f"Sampled {total_flips} transitions in {time.time()-stime:.3f}s")

# Empirical estimate 
stime = time.time()
sigma_emp = spin_model.get_empirical_EP(beta, J, S, F)
time_emp  = time.time() - stime

# Running sums to keep track of EP estimates
sigma_g = sigma_g2 = sigma_N1 = sigma_MTUR = 0.0

# Running sums to keep track of EP estimates
time_g  = time_g2 = time_N1 = time_MTUR = 0.0

# Because system is multipartite, we can separately estimate EP for each spin
for i in tqdm(range(N), smoothing=0):
    p_i            =  F[:,i].sum()/total_flips               # frequency of spin i flips

    # Calculate samples of g observables for states in which spin i changes state
    stime = time.time()
    g_samples               = utils.numpy_to_torch(spin_model.get_g_observables(S, F, i))
    time_MTUR += time.time() - stime
#    g_mean                  = g_samples.mean(axis=0)

    do_holdout = False
    if not test_old:
        stime = time.time()
        data = epm.Dataset(g_samples=g_samples)
        # Create dataset with holdout data
        # Full optimization with trust-region Newton method and holdout 
        if do_holdout:
            train, test = data.split_train_test(holdout_shuffle=False)
            spin_full = epm.get_EP_Newton(train, trust_radius=1/4, holdout_data=test).objective
        else:
            spin_full = epm.get_EP_Newton(data).objective
        time_g  += time.time() - stime

        sigma_g    += p_i * spin_full
        # # Full optimization with gradient ascent method and holdout
        # stime = time.time()
        # spin_grad = ep_estimators.get_EP_GradientAscent(train, holdout_data=test).objective
        # time_g2 += time.time() - stime

    else:
        # Create dataset with holdout data
        # Full optimization with trust-region Newton method and holdout 
        stime = time.time()
        obj = epm.EPEstimators(g_samples=g_samples) 
        spin_grad = obj.get_EP_Newton(trust_radius=1/4, holdout=do_holdout).objective
        time_g2 += time.time() - stime
        sigma_g2   += p_i * spin_grad



    utils.empty_torch_cache()


print(f"\nEntropy production estimates (N={N}, k={k}, β={beta})")
print(f"  Σ     (Empirical)                         :    {sigma_emp :.6f}  ({time_emp :.3f}s)")
print(f"  Σ_g   (Full optimization w/ Newton)       :    {sigma_g   :.6f}  ({time_g   :.3f}s)")
print(f"  Σ_g   (Full optimization w/ grad. ascent) :    {sigma_g2  :.6f}  ({time_g2  :.3f}s)")
print(f"  Σ̂_g   (1-step Newton)                     :    {sigma_N1  :.6f}  ({time_N1  :.3f}s)")
print(f"  Σ_TUR (Multidimensional MTUR)             :    {sigma_MTUR:.6f}  ({time_MTUR:.3f}s)")

