import time, os
import numpy as np
from tqdm import tqdm

import spin_model
import observables
import ep_estimators
import utils

# The following allows torch to use GPU for computation, if available
utils.set_default_torch_device()                

# Setup model parameters for the nonequilibrium spin model (asymmetric kinetic Ising model)
N    = 10    # system size
k    = 6     # avg number of neighbors in sparse coupling matrix
beta = 2.0   # inverse temperature

np.random.seed(42)                                                          # Set seed for reproducibility
stime = time.time()
J    = spin_model.get_couplings_random(N=N, k=k)                            # Generate coupling matrix
S, F = spin_model.run_simulation(beta=beta, J=J, samples_per_spin=100000)   # Run Monte Carlo simulation
num_samples_per_spin, N = S.shape
total_flips = N * num_samples_per_spin                                      # Total spin-flip attempts
print(f"Sampled {total_flips} transitions in {time.time()-stime:.3f}s")

# Ground truth estimate of EP based on empirical statistics
stime = time.time()
sigma_emp = spin_model.get_empirical_EP(beta, J, S, F)
time_emp  = time.time() - stime

# Running sums to keep track of EP estimates
sigma_g = sigma_g2 = sigma_N1 = sigma_MTUR = 0.0

# Running sums to keep track of time taken for each estimator
time_g  = time_g2 = time_N1 = time_MTUR = 0.0

# Because system and observables are multipartite, we can separately estimate EP for each spin
for i in tqdm(range(N), smoothing=0):
    p_i           =  F[:,i].sum()/total_flips               # frequency of spin i flips

    # Calculate samples of g observables for states in which spin i changes state
    g_samples     = observables.get_g_observables(S, F, i)
    g_mean        = g_samples.mean(axis=0)

    data          = observables.Dataset(g_samples=g_samples)

    # Create dataset with validation and test holdout data
    train, val, test = data.split_train_val_test()

    # Full optimization with validation dataset (for early stopping) and test set (for evaluating the objective)
    # By default, we use gradient ascent with Barzilai-Borwein step sizes
    stime        = time.time()
    spin_g, _    = ep_estimators.get_EP_Estimate(data, validation=val, test=test)
    sigma_g     += p_i * spin_g 
    time_g      += time.time() - stime

    # 1 step of Newton
    stime        = time.time()
    spin_N1, _   = ep_estimators.get_EP_Newton1Step(data, validation=val, test=test) 
    time_N1     += time.time() - stime
    sigma_N1    += p_i * spin_N1

    # Multidimensional TUR
    stime         = time.time()
    spin_MTUR, _  = ep_estimators.get_EP_MTUR(data)
    time_MTUR    += time.time() - stime
    sigma_MTUR   += p_i * spin_MTUR
    

print(f"\nEntropy production estimates (N={N}, k={k}, β={beta})")
print(f"  Σ     (Empirical)             :    {sigma_emp :.6f}  ({time_emp :.3f}s)")
print(f"  Σ_g   (Full optimization)     :    {sigma_g   :.6f}  ({time_g   :.3f}s)")
print(f"  Σ̂_g   (1-step Newton)         :    {sigma_N1  :.6f}  ({time_N1  :.3f}s)")
print(f"  Σ_TUR (Multidimensional MTUR) :    {sigma_MTUR:.6f}  ({time_MTUR:.3f}s)")

