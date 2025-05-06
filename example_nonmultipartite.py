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
beta = .5   # inverse temperature


np.random.seed(42) # Set seed for reproducibility

stime = time.time()
J    = spin_model.get_couplings_random(N=N, k=k)
S, F = spin_model.run_simulation(beta=beta, J=J, samples_per_spin=10000)
print(f"Ran Monte Carlo in {time.time()-stime:.3f}s")

X0, X1 = spin_model.convert_to_nonmultipartite(S, F)

# Empirical estimate 
stime = time.time()
sigma_emp = spin_model.get_empirical_EP(beta, J, S, F)
time_emp  = time.time() - stime



data          = ep_estimators.RawDataset(X0, X1)
estimator_obj = ep_estimators.EPEstimators(data)


# Full optimization with gradient ascent method 
stime = time.time()
sigma_g2 = estimator_obj.get_EP_GradientAscent(holdout=True).objective
time_g2  = time.time() - stime

# # Multidimensional TUR
# stime = time.time()
# sigma_MTUR = epm.get_EP_MTUR(g_samples=g_samples, rev_g_samples=-g_samples)             
# time_MTUR = time.time() - stime

print(f"\nEntropy production estimates")
print(f"  Σ     (Empirical)                         :    {sigma_emp :.6f}  ({time_emp :.3f}s)")
print(f"  Σ_g   (Full optimization w/ grad. ascent) :    {sigma_g2  :.6f}  ({time_g2  :.3f}s)")
# print(f"  Σ_TUR (Multidimensional MTUR)             :    {sigma_MTUR:.6f}  ({time_MTUR:.3f}s)")
