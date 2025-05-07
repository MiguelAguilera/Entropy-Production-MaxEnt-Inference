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

N    = 200     # system size
k    = 6      # avg number of neighbors in sparse coupling matrix
beta = 3      # inverse temperature


np.random.seed(42) # Set seed for reproducibility

stime = time.time()
J    = spin_model.get_couplings_random(N=N, k=k)
S, F = spin_model.run_simulation(beta=beta, J=J, samples_per_spin=1500)

# Empirical estimate 
stime = time.time()
sigma_emp = spin_model.get_empirical_EP(beta, J, S, F)
time_emp  = time.time() - stime

X0, X1 = spin_model.convert_to_nonmultipartite(S, F)
print(f"Ran Monte Carlo in {time.time()-stime:.3f}s, get {X0.shape[0]} samples")
print()


stime = time.time()
data1       = ep_estimators.RawDataset(X0, X1)
trn1, tst1  = data1.split_train_test()
print("Running get_EP_GradientAscent with RawDataset")
solutionS1  = ep_estimators.get_EP_GradientAscent(data=trn1, holdout_data=tst1, max_iter=5000, lr=5e-1, tol=1e-8, use_Adam=False, verbose=2)
time_S1     = time.time() - stime


data2       = ep_estimators.RawDataset2(X0, X1)
trn2, tst2  = data2.split_train_test()
print("Running get_EP_GradientAscent with RawDataset2")
solutionS2  = ep_estimators.get_EP_GradientAscent(data=trn2, holdout_data=tst2, max_iter=5000, lr=5e-1, tol=1e-8, use_Adam=False, verbose=2)
time_S2     = time.time() - stime


print(f"\nEntropy production estimates (N={N}, k={k}, β={beta})")
print(f"  Σ     (Empirical)                                     :                  {sigma_emp :.6f}  ({time_emp :.3f}s)")
print(f"  Σ_g   (Using antisymmetric observables, grad. ascent) :    tst objective={solutionS1.objective:6f}  ({time_S1  :.3f}s)")
print(f"  Σ_g   (Using full observables         , grad. ascent) :    tst objective={solutionS2.objective:6f}  ({time_S2  :.3f}s)")



