import time, sys
import numpy as np

sys.path.insert(0, '..')
import spin_model
import ep_estimators
import observables
import torch


import argparse
parser = argparse.ArgumentParser(description="Nonequilibrium spin model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--N", type=int, default=10, help="System size")
parser.add_argument("--k", type=int, default=6, help="Avg number of neighbors in sparse coupling matrix")
parser.add_argument("--beta", type=float, default=2.0, help="Inverse temperature")
parser.add_argument("--samples_per_spin", type=int, default=100000, help="Samples per spin")
parser.add_argument("--NEEP", action="store_true", default=False, help="Use NEEP objective")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

np.random.seed(args.seed)

if args.NEEP:
    import NEEP
    print("Using NEEP objective")
    dataset_class = NEEP.DatasetNEEP
else:
    dataset_class = observables.Dataset

stime = time.time()
J    = spin_model.get_couplings_random(N=args.N, k=args.k)
S, F = spin_model.run_simulation(beta=args.beta, J=J, samples_per_spin=args.samples_per_spin, seed=args.seed)
N    = J.shape[0]
X0, X1 = spin_model.convert_to_nonmultipartite(S, F)
print(f"Ran Monte Carlo in {time.time()-stime:.3f}s, get {X0.shape[0]} samples")


# The following allows torch to use GPU for computation, if available
# However, for small examples, it can be faster to use CPU
import utils
utils.set_default_torch_device()



# Empirical estimate 
stime     = time.time()
sigma_emp = spin_model.get_empirical_EP(args.beta, J, S, F)
time_emp  = time.time() - stime

print(f"\nEntropy production estimates (N={args.N}, k={args.k}, β={args.beta})")
print(f"  Σ     (Empirical)                                :    {sigma_emp      :.6f}  ({time_emp      :.3f}s)")

for observable_ix, observable_desc in enumerate(["x'ᵢxⱼ−xⱼ'xᵢ", "(x'ᵢ−xᵢ)xⱼ"]):
    # Calculate antisymmetric observables explicitly
    if observable_ix == 0:
        g_samples = np.vstack([ X1[:,i]*X0[:,j] - X0[:,i]*X1[:,j] 
                                for i in range(N) for j in range(i+1, N) ]).T
        g_samples = np.vstack([ (X1[:,i]+X1[:,j])*X0[:,j] - (X0[:,i]+X0[:,j])*X1[:,j]
                                for i in range(N) for j in range(i+1, N) ]).T
        #dataS     = observables.CrossCorrelations1(X0, X1)
    else:
        # Calculate samples of g observables for states in which spin i changes state
        g_samples = np.vstack([ (X1[:,i] - X0[:,i])*X0[:,j] 
                                for i in range(N) for j in range(N) if i != j]).T

        #X0 = X0.astype(np.float32)
        #X1 = X1.astype(np.float32)
        #dataS     = observables.CrossCorrelations2(X0, X1)

    data             = dataset_class(g_samples=g_samples)

    #np.random.seed(42) # Set seed for reproducibility of holdout shuffles
    #trainS, valS, testS = dataS.split_train_val_test()
    np.random.seed(42) # Set seed for reproducibility of holdout shuffles
    train, val, test = data.split_train_val_test()

    stime            = time.time()
    sigma_N_obs, _   = ep_estimators.get_EP_Newton1Step(train, validation=val, test=test)
    time_N_obs       = time.time() - stime


    stime = time.time()
    sigma_G_obs, _   = ep_estimators.get_EP_Estimate(train, validation=val, test=test)
    time_G_obs       = time.time() - stime


    # # Estimate EP using gradient ascent method , from state samples
    # stime = time.time()
    # sigma_S_obs, _   = ep_estimators.get_EP_Estimate(trainS, validation=valS, test=testS)
    # time_S_obs       = time.time() - stime

    print(f"Observables gᵢⱼ(x) = {observable_desc}")
    # print(f"  Σ_g   (From state samples     , gradient ascent) :    {sigma_S_obs :.6f}  ({time_S_obs  :.3f}s)")
    print(f"  Σ_g   (From observable samples, gradient ascent) :    {sigma_G_obs :.6f}  ({time_G_obs    :.3f}s)")
    print(f"  Σ̂_g   (From observable samples, 1-step Newton  ) :    {sigma_N_obs :.6f}  ({time_N_obs    :.3f}s)")

    torch.mps.empty_cache()