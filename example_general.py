import time, os
import numpy as np
from tqdm import tqdm

import spin_model
import ep_estimators2 as ep_estimators
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
N    = J.shape[0]

# Empirical estimate 
stime     = time.time()
sigma_emp = spin_model.get_empirical_EP(beta, J, S, F)
time_emp  = time.time() - stime

X0, X1 = spin_model.convert_to_nonmultipartite(S, F)
print(f"Ran Monte Carlo in {time.time()-stime:.3f}s, get {X0.shape[0]} samples")

print(f"\nEntropy production estimates (N={N}, k={k}, β={beta})")
print(f"  Σ     (Empirical)                                :    {sigma_emp      :.6f}  ({time_emp      :.3f}s)")

for observable_ix, observable_desc in enumerate(["x'ᵢxⱼ−xⱼ'xᵢ", "(x'ᵢ−xᵢ)xⱼ"]):
    # Calculate antisymmetric observables explicitly
    if observable_ix == 0:
        g_samples = np.vstack([ X1[:,i]*X0[:,j] - X0[:,i]*X1[:,j] 
                                for i in range(N) for j in range(i+1, N) ]).T
        trainS, valS, testS = ep_estimators.DatasetStateSamples(X0, X1).split_train_val_test()
    else:
        # Calculate samples of g observables for states in which spin i changes state
        g_samples = np.vstack([ (X1[:,i] - X0[:,i])*X1[:,j] 
                                for i in range(N) for j in range(N) if i != j]).T
        trainS, valS, testS = ep_estimators.DatasetStateSamples2(X0, X1).split_train_val_test()

    g_mean = g_samples.mean(axis=0)

    data1            = ep_estimators.Dataset(g_samples=g_samples)
    train, val, test = data1.split_train_val_test()

    stime            = time.time()
    sigma_N_obs, _   = ep_estimators.get_EP_Newton1Step(train, validation=val, test=test)
    time_N_obs       = time.time() - stime


    stime = time.time()
    sigma_G_obs, _   = ep_estimators.get_EP_Estimate(train, validation=val, test=test)
    time_G_obs       = time.time() - stime

    # Estimate EP using gradient ascent method , from state samples
    stime = time.time()
    sigma_S_obs, _   = ep_estimators.get_EP_Estimate(trainS, validation=valS, test=testS)
    time_S_obs       = time.time() - stime

    print(f"Observables gᵢⱼ(x) = {observable_desc}")
    print(f"  Σ_g   (From observable samples, gradient ascent) :    {sigma_G_obs :.6f}  ({time_G_obs    :.3f}s)")
    print(f"  Σ̂_g   (From observable samples, 1-step Newton  ) :    {sigma_N_obs :.6f}  ({time_N_obs    :.3f}s)")
    print(f"  Σ_g   (From state samples, gradient ascent     ) :    {sigma_S_obs :.6f}  ({time_S_obs  :.3f}s)")

