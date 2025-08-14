import time, os, sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '..')

import spin_model
import observables
import ep_estimators
from utils import numpy_to_torch

import argparse
# Also show default values for the options
parser = argparse.ArgumentParser(description="Nonequilibrium spin model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--N", type=int, default=10, help="System size")
parser.add_argument("--k", type=int, default=6, help="Avg number of neighbors in sparse coupling matrix")
parser.add_argument("--beta", type=float, default=2.0, help="Inverse temperature")
parser.add_argument("--samples_per_spin", type=int, default=100000, help="Samples per spin")
parser.add_argument("--NEEP", action="store_true", default=False, help="Use NEEP objective")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--obs_subset", type=int, default=0, help="Number of observables per spin to use (0 for all)")
args = parser.parse_args()

if args.NEEP:
    import NEEP
    print("Using NEEP objective")
    dataset_class = NEEP.DatasetNEEP
else:
    dataset_class = observables.Dataset


def get_g_observables_restricted(S, F, i, k):
    # Given states S ∈ {-1,1} in which spin i flipped, we calculate the observables 
    #        g_{ij}(x) = (x_i' - x_i) x_j 
    # for all j, where x_i' and x_i indicate the state of spin i after and before the jump, 
    # and x_j is the state of every other spin.
    # Here F indicates which spins flipped. We try to use GPU and in-place operations
    # where possible, to increase speed while reducing GPU memory requirements

    # here we restrict to only include the first k observables
    S_i      = S[F[:,i],:]
    X        = numpy_to_torch(np.delete(S_i, i, axis=1))
    y        = numpy_to_torch(S_i[:,i])
    X.mul_( (-2 * y).unsqueeze(1) )  # in place multiplication
    if k > 0:
        X = X[:,:k]
    return X.contiguous()




np.random.seed(args.seed)
stime = time.time()
J    = spin_model.get_couplings_random(N=args.N, k=args.k)                                       # Generate coupling matrix
S, F = spin_model.run_simulation(beta=args.beta, J=J, samples_per_spin=args.samples_per_spin)    # Run Monte Carlo simulation
num_samples_per_spin, N = S.shape
total_flips = N * num_samples_per_spin                                      # Total spin-flip attempts
print(f"Sampled {total_flips} transitions in {time.time()-stime:.3f}s")

# The following allows torch to use GPU for computation, if available
# However, for small examples, it can be faster to use CPU
import utils
utils.set_default_torch_device()

# Ground truth estimate of EP based on empirical statistics
stime = time.time()
sigma_emp = spin_model.get_empirical_EP(args.beta, J, S, F)
time_emp  = time.time() - stime

# Running sums to keep track of EP estimates
sigma_g = sigma_g2 = sigma_N1 = sigma_MTUR = 0.0

# Running sums to keep track of time taken for each estimator
time_g  = time_g2 = time_N1 = time_MTUR = 0.0

# Because system and observables are multipartite, we can separately estimate EP for each spin
for i in tqdm(range(N), smoothing=0):
    p_i           =  F[:,i].sum()/total_flips               # frequency of spin i flips

    # Calculate samples of g observables for states in which spin i changes state
    g_samples     = get_g_observables_restricted(S, F, i, k=args.obs_subset)

    data          = dataset_class(g_samples=g_samples)

    # Create dataset with validation and test holdout data
    train, val, test = data.split_train_val_test()

    # Full optimization with validation dataset (for early stopping) and test set (for evaluating the objective)
    # By default, we use gradient ascent with Barzilai-Borwein step sizes
    stime        = time.time()
    spin_g, _    = ep_estimators.get_EP_Estimate(data, validation=val, test=test)
    sigma_g     += p_i * spin_g 
    time_g      += time.time() - stime

    # Full optimization with validation dataset (for early stopping) and test set (for evaluating the objective)
    # Here is an example of how to use a different optimizer
    stime        = time.time()
    spin_g2, _   = ep_estimators.get_EP_Estimate(data, validation=val, test=test, optimizer='NewtonMethodTrustRegion')
    sigma_g2    += p_i * spin_g2
    time_g2     += time.time() - stime

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
    

print(f"\nEntropy production estimates (N={args.N}, k={args.k}, β={args.beta})")
print(f"  Σ     (Empirical)                        :    {sigma_emp :.6f}  ({time_emp :.3f}s)")
print(f"  Σ_g   (Full optimization, gradient asc.) :    {sigma_g   :.6f}  ({time_g   :.3f}s)")
print(f"  Σ_g   ( ... trust region Newton method)  :    {sigma_g2  :.6f}  ({time_g2  :.3f}s)")
print(f"  Σ̂_g   (1-step Newton method)             :    {sigma_N1  :.6f}  ({time_N1  :.3f}s)")
print(f"  Σ_TUR (Multidimensional TUR)             :    {sigma_MTUR:.6f}  ({time_MTUR:.3f}s)")

