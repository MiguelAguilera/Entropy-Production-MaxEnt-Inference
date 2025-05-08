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
beta = 2      # inverse temperature


np.random.seed(42) # Set seed for reproducibility

stime = time.time()
J    = spin_model.get_couplings_random(N=N, k=k)
S, F = spin_model.run_simulation(beta=beta, J=J, samples_per_spin=10000)

# Empirical estimate 
stime = time.time()
sigma_emp = spin_model.get_empirical_EP(beta, J, S, F)
time_emp  = time.time() - stime

X0, X1 = spin_model.convert_to_nonmultipartite(S, F)
print(f"Ran Monte Carlo in {time.time()-stime:.3f}s, get {X0.shape[0]} samples")


# Estimate EP using gradient ascent method , from observable samples
N = J.shape[0]
# Calculate antisymmetric observables explicitly
g_samples = np.vstack([ X1[:,i]*X0[:,j] - X0[:,i]*X1[:,j] 
                        for i in range(N) for j in range(i+1, N) ]).T
# Calculate antisymmetric observables explicitly
g_samples = np.vstack([ (X1[:,i] - X0[:,i])*X1[:,j] 
                        for i in range(N) for j in range(N) if i != j]).T

data1         = ep_estimators.Dataset(g_samples=g_samples)
train, test   = data1.split_train_test() 

stime = time.time()
sigma_N_obs  = ep_estimators.get_EP_Newton(train, holdout_data=test, trust_radius=1).objective
time_N_obs   = time.time() - stime


stime = time.time()
sigma_G_obs  = ep_estimators.get_EP_GradientAscent(train, holdout_data=test).objective
time_G_obs   = time.time() - stime

# Estimate EP using gradient ascent method , from state samples
stime = time.time()
trainS, testS = ep_estimators.RawDataset2(X0, X1).split_train_test()
sigma_S_state = ep_estimators.get_EP_GradientAscent(train, holdout_data=test).objective
time_S_state  = time.time() - stime


print(f"\nEntropy production estimates (N={N}, k={k}, β={beta})")
print(f"  Σ     (Empirical)                              :    {sigma_emp :.6f}  ({time_emp :.3f}s)")
print(f"  Σ_g   (Using observable samples, grad. ascent) :    {sigma_G_obs    :.6f}  ({time_G_obs    :.3f}s)")
print(f"  Σ_g   (Using observable samples, Newton method):    {sigma_N_obs    :.6f}  ({time_N_obs    :.3f}s)")
print(f"  Σ_g   (Using state samples, grad. ascent)      :    {sigma_S_state  :.6f}  ({time_S_state  :.3f}s)")

