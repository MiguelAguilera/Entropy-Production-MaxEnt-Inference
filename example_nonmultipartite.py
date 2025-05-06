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

N    = 10     # system size
k    = 6      # avg number of neighbors in sparse coupling matrix
beta = 3.25   # inverse temperature


np.random.seed(42) # Set seed for reproducibility

stime = time.time()
J    = spin_model.get_couplings_random(N=N, k=k)
S, F = spin_model.run_simulation(beta=beta, J=J, samples_per_spin=10000)
print(f"Ran Monte Carlo in {time.time()-stime:.3f}s")

# Empirical estimate 
stime = time.time()
sigma_emp = spin_model.get_empirical_EP(beta, J, S, F)
time_emp  = time.time() - stime

X0, X1 = spin_model.convert_to_nonmultipartite(S, F)


# Estimate EP using gradient ascent method , from observable samples
N = J.shape[0]
# Calculate antisymmetric observables explicitly
g_samples = np.vstack([ X1[:,i]*X0[:,j] - X0[:,i]*X1[:,j] 
                        for i in range(N) for j in range(i+1, N) ]).T
stime = time.time()
data1         = ep_estimators.Dataset(g_mean=g_samples.mean(axis=0), rev_g_samples=-g_samples)
estimator1    = ep_estimators.EPEstimators(data1)
sigma_g2_obs  = estimator1.get_EP_GradientAscent(holdout=True).objective
time_g2_obs   = time.time() - stime


# Estimate EP using gradient ascent method , from state samples
stime = time.time()
data2          = ep_estimators.RawDataset(X0, X1)
estimator2     = ep_estimators.EPEstimators(data2)
sigma_g2_state = estimator2.get_EP_GradientAscent(holdout=True).objective
time_g2_state  = time.time() - stime


print(f"\nEntropy production estimates (N={N}, k={k}, β={beta})")
print(f"  Σ     (Empirical)                              :    {sigma_emp :.6f}  ({time_emp :.3f}s)")
print(f"  Σ_g   (Using observable samples, grad. ascent) :    {sigma_g2_obs    :.6f}  ({time_g2_obs    :.3f}s)")
print(f"  Σ_g   (Using state samples, grad. ascent)      :    {sigma_g2_state  :.6f}  ({time_g2_state  :.3f}s)")

theta = np.random.rand(data1.nobservables)
print(data1.get_objective(theta))
print(data2.get_objective(theta))

print(data1.get_tilted_statistics(theta=theta, return_mean=True))
print(data2.get_tilted_statistics(theta=theta, return_mean=True))
