import time, os
import numpy as np
from tqdm import tqdm

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable torch fallback for MPS backend
import torch

import spin_model
import ep_estimators
import utils

# The following allows torch to use GPU for computation
utils.set_default_torch_device()                

N    = 10   # system size
k    = 6    # avg number of neighbors in sparse coupling matrix
beta = 2.0   # inverse temperature

np.random.seed(42) # Set seed for reproducibility

stime = time.time()
J    = spin_model.get_couplings_random(N=N, k=k)
S, F = spin_model.run_simulation(beta=beta, J=J, samples_per_spin=100000)
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
    g_samples               = utils.numpy_to_torch(spin_model.get_g_observables(S, F, i))
    g_mean                  = g_samples.mean(axis=0)

    data = ep_estimators.Dataset(g_samples=g_samples)

    # 1 step of Newton
    stime       = time.time()
    spin_N1     = ep_estimators.get_EP_Newton(data, max_iter=1).objective 
    time_N1    += time.time() - stime
    sigma_N1   += p_i * spin_N1

    # Multidimensional TUR
    stime       = time.time()
    spin_MTUR   = ep_estimators.get_EP_MTUR(data).objective
    time_MTUR  += time.time() - stime
    sigma_MTUR += p_i * spin_MTUR
    
    # Create dataset with holdout data
    train, test = data.split_train_test(holdout_shuffle=True)
    # Full optimization with trust-region Newton method and holdout 
    stime      = time.time()
    sol_newton = ep_estimators.get_EP_Newton(train, trust_radius=1/4, holdout_data=test)
    sigma_g   += p_i * sol_newton.objective 
    time_g    += time.time() - stime

    # Full optimization with gradient ascent method and holdout
    stime     = time.time()
    spin_grad = ep_estimators.get_EP_GradientAscent(train, holdout_data=test).objective
    time_g2   += time.time() - stime
    sigma_g2  += p_i * spin_grad

    utils.empty_torch_cache()


print(f"\nEntropy production estimates (N={N}, k={k}, β={beta})")
print(f"  Σ     (Empirical)                         :    {sigma_emp :.6f}  ({time_emp :.3f}s)")
print(f"  Σ_g   (Full optimization w/ Newton)       :    {sigma_g   :.6f}  ({time_g   :.3f}s)")
print(f"  Σ_g   (Full optimization w/ grad. ascent) :    {sigma_g2  :.6f}  ({time_g2  :.3f}s)")
print(f"  Σ̂_g   (1-step Newton)                     :    {sigma_N1  :.6f}  ({time_N1  :.3f}s)")
print(f"  Σ_TUR (Multidimensional MTUR)             :    {sigma_MTUR:.6f}  ({time_MTUR:.3f}s)")

