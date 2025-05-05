import time, os
import numpy as np
from tqdm import tqdm

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable torch fallback for MPS backend
import torch

import spin_model
import ep_estimators as epm
import utils

# The following allows torch to use GPU for computation
utils.set_default_torch_device()                

N    = 10   # system size
k    = 6    # avg number of neighbors in sparse coupling matrix
beta = .5   # inverse temperature


np.random.seed(42) # Set seed for reproducibility

stime = time.time()
J    = spin_model.get_couplings_random(N=N, k=k)
S, F = spin_model.run_simulation(beta=beta, J=J, samples_per_spin=10000)
num_samples_per_spin, N = S.shape
total_flips = N * num_samples_per_spin  # Total spin-flip attempts
print(f"Samples {total_flips} transitions in {time.time()-stime:.3f}s")

# Running sums to keep track of EP estimates
sigma_emp = sigma_g = sigma_g2 = sigma_N1 = sigma_MTUR = 0.0

stime = time.time()

# Because system is multipartite, we can separately estimate EP for each spin
for i in tqdm(range(N)):
    p_i            =  F[:,i].sum()/total_flips               # frequency of spin i flips

    # Calculate samples of g observables for states in which spin i changes state
    g_samples               = utils.numpy_to_torch(spin_model.get_g_observables(S, F, i))
    g_mean                  = g_samples.mean(axis=0)

    # Calculate empirical estimate of true EP (contribution from i-th spin)

    spin_emp = spin_model.get_spin_empirical_EP(beta, J, i, g_mean)
    
    obj = epm.EPEstimators(g_mean=g_mean, rev_g_samples=-g_samples)

    # 1 step of Newton
    spin_N1   = obj.get_EP_Newton(max_iter=1).objective 
    
    # Full optimization with trust-region Newton method and holdout 
    spin_full = obj.get_EP_Newton(trust_radius=1/4, holdout=True).objective

    # Full optimization with gradient ascent method 
    spin_grad = obj.get_EP_GradientAscent(holdout=True).objective

    # Multidimensional TUR
    spin_MTUR = epm.get_EP_MTUR(g_samples=g_samples, rev_g_samples=-g_samples)             
    
    sigma_emp  += p_i * spin_emp
    sigma_g    += p_i * spin_full
    sigma_N1   += p_i * spin_N1
    sigma_MTUR += p_i * spin_MTUR
    sigma_g2   += p_i * spin_grad

    utils.empty_torch_cache()


print(f"\nEntropy production estimates (took {time.time()-stime:3f}s):")
print(f"  Σ     (Empirical)                         :    {sigma_emp:.6f}")
print(f"  Σ_g   (Full optimization w/ Newton)       :    {sigma_g:.6f}")
print(f"  Σ_g   (Full optimization w/ grad. ascent) :    {sigma_g2:.6f}")
print(f"  Σ̂_g   (1-step Newton)                     :    {sigma_N1:.6f}")
print(f"  Σ_TUR (Multidimensional MTUR)             :    {sigma_MTUR:.6f}")

