# Nonequilibrium spin model used in the manuscript:
#  Aguilera, Ito, Kolchinsky, Inferring entropy production in many-body 
#                             systems using nonequilibrium MaxEnt
#
# We include code to generate random coupling matrices and to perform Monte Carlo 
# simulations using Glauber dynamics. Here we use the convention that the coupling 
# matrix is indicated as J (not w as in the manuscript)

import numpy as np
from numba import njit, objmode, prange

from config import DTYPE   # Default data type for numerical operations

# ***** Methods to generate coupling matrices *****
def get_couplings_random(N, k=None, J0=0.0, DJ=1.0):
    """
    Generate coupling matrix for (possibly) diluted nonequilibrium spin model. 

    See for example: Erik Aurell and Hamed Mahmoudi J. Stat. Mech. (2011) P04014

    Args:
        N (int)    : Number of spins
        k (int)    : Sparsity parameter for diluted model (~= mean number of neighbors).
                     If None, then return the dense model with all-to-all connectivity
        J0 (float) : Mean coupling
        DJ (float) : Scales coupling standard deviations variability
    """
    rnd = np.random.randn(N, N)

    if k is not None:
        assert(k >= 0)
        mask = np.random.rand(N, N) < k/(N-1)
        rnd *= mask
        norm_const = np.sqrt(k)
    else:
        norm_const = np.sqrt(N)

    J = (J0 / N + rnd * DJ / norm_const).astype(DTYPE)
    np.fill_diagonal(J, 0)
    return J




# ***** Monte Carlo code  *****

@njit(inline='always')
def GlauberStep(i, Ji, s, N_s=100):
    """
    Perform a single Glauber update for a continuous spin (angle in [0, 2π)) using a discretized Gibbs sampler.

    Arguments:
        i       : Index of the spin to update
        Ji      : Couplings from spin i to others
        s       : Current spin configuration (angles in radians)
        N_s : Number of discretized angles to sample from

    Returns:
        float   : New angle (in [0, 2π)) for spin i
    """
    ds = 2 * np.pi / N_s
    probs = np.empty(N_s, dtype=DTYPE)

    for k in range(N_s):
        s_new = k * ds
        energy = -np.sum(Ji * np.cos(s_new - s))
        probs[k] = np.exp(-energy)

    # Normalize
    Z = np.sum(probs)
    probs /= Z

    # Sample with interpolation
    r = np.random.rand()
    cdf = 0.0
    for k in range(N_s):
        prev_cdf = cdf
        cdf += probs[k]
        if r < cdf:
            if k == 0:
                frac = r / cdf
                return frac * ds
            else:
                # Linear interpolation between (k-1) and k
                interval_prob = cdf - prev_cdf
                frac = (r - prev_cdf) / interval_prob
                return ((k - 1) + frac) * ds

    # Fallback
    return (N_s - 1) * ds

@njit(f'{DTYPE}[:]({DTYPE}[:,::1], {DTYPE}[::1], int32)', inline='always')
def ParallelGlauberStep(J, s, T=1):
    """
    Parallel Glauber dynamics: all spins are updated simultaneously.

    Args:
        J (matrix): Coupling matrix.
        s (array): Initial spin state.
        T (int): Number of Monte Carlo sweeps.

    Returns:
        array: Final spin configuration.
    """
    size = len(s)
    for t in range(T):
        s_p = s.copy()
        for i in range(size):
            s[i] = GlauberStep(i, J[i, :], s_p)
    return s


@njit(parallel=True, fastmath=True)
def run_simulation(beta, J, warmup=0.1, samples_per_spin=1_000_000, thinning_multiplier=1,
                   num_restarts=1000, sequential=True, progressbar=True, seed=None):
    """
    Monte Carlo sampling of nonequilibrium spin model using Glauber dynamics.
    By default, we use a thinning factor of N between samples

    Args:
        beta (float)                : Inverse temperature
        J (2d np.array)             : NxN matrix of coupling coefficients
        warmup (float)              : Fraction of Monte Carlo steps in the warm-up at the beginning of each restart
        samples_per_spin (int)      : Number of samples per spin to return
        thinning_multiplier (int)   : In between these samples, we reduce correlations by discarding 
                                       thinning_multiplier * N samples
        num_restarts (int)          : Number of times to restart sampler
        sequential (bool)           : Whether to use sequential or parallel updates
        progressbar (bool)          : Whether to display progressbar during simulation
        seed (int)                  : Random seed for reproducibility

    Returns:
        S: Nxsamples_per_spin int array : Samples of -1,1 state from stationary distribution
        F: Nxsamples_per_spin bool array : Samples of state transitions (True: flipped, False: no flip)
    """

    N = J.shape[0]
    betaJ = (beta*J).astype(DTYPE)

    samples_per_spin = int(samples_per_spin)

    # Define matrix of spin states and spin flips      
    S = np.empty((samples_per_spin, N), dtype=DTYPE)
    S1 = np.empty((samples_per_spin, N), dtype=DTYPE)

    samples_per_restart = samples_per_spin//num_restarts
    
    print_every = int(samples_per_spin/100)
    if progressbar:
        print("-"*100)

    for restart_ix in prange(num_restarts):
        if seed is not None:
            np.random.seed(seed + restart_ix)

        # Start from a random state, then warm up for N * warmup * samples_per_restart steps
#        s = ((np.random.randint(0, 2, N) * 2) - 1).astype(DTYPE)
        s = (2 * np.pi * np.random.rand(N)).astype(DTYPE)  # angles in [0, 2π)
        if sequential:
            indices = np.random.randint(0, N, int(N * warmup * samples_per_restart))
            for i in indices:
                s[i] = GlauberStep(i, betaJ[i, :], s)
        else:
            s = ParallelGlauberStep(betaJ, s, T=int(samples_per_restart * warmup))

        # Now draw samples from steady state
        for r in range(samples_per_restart):
            # First, thin out samples by N steps to decrease correlations
            if sequential:
                indices = np.random.randint(0, N, int(N*thinning_multiplier))
                for i in indices:
                    s[i] = GlauberStep(i, betaJ[i, :], s)

            else:
                s = ParallelGlauberStep(betaJ, s, T=thinning_multiplier)
            
            # Save the sampled state
            sample_ix = restart_ix * samples_per_restart + r
            S[sample_ix, :] = s

            # Sample possible flips for each spoin
            s1 = np.empty(N, dtype=DTYPE)
            for i in range(N):
                s1[i] = GlauberStep(i, betaJ[i, :], s)

            # Save the updated state
            S1[sample_ix, :] = s1

            if progressbar and sample_ix % print_every == 0:
                with objmode():
                    print(".", end="", flush=True)

    if progressbar:
        print()

    return S, S1


#def convert_to_nonmultipartite(S, F):
#    # Convert samples S and F to non-multipartite form
#    # Arguments:
#    #   S: samples_per_spin x N (int)  : initial state ∈ {-1,1}^N at beginning of each of samples_per_spin x N transition attempts
#    #   F: samples_per_spin x N (bool) : whether each spin flipped or not in each of samples_per_spin x N transition attempts
#    # Returns:
#    #   X0: (samples_per_spin x N) x N : initial states for each spin attempt
#    #   X1: (samples_per_spin x N) x N : final states for each spin attempt

#    # Stack S and F into non-multipartite datastructure
#    X0s, X1s = [], []
#    N = S.shape[1]
#    for i in range(N):
#        X0s.append(S)
#        Sp = S.copy()
#        Sp[F[:,i],i] *= -1
#        X1s.append(Sp)
#    X0 = np.vstack(X0s)
#    X1 = np.vstack(X1s)

#    # Shuffle data
#    perm = np.random.permutation(X0.shape[0])
#    return X0[perm], X1[perm]



# # ***** Entropy production estimates from empirical statistics *****

# import observables
# import utils
# def get_empirical_EP(beta, J, S, S1):
#    # Calculate empirical EP from samples S and F
#    num_samples_per_spin, N = S.shape
#    total_flips = N * num_samples_per_spin  # Total spin-flip attempts

#    # Running sums to keep track of EP estimates
#    sigma_emp = 0.0

#    # Because system is multipartite, we can separately estimate EP for each spin

#    for i in range(N):
#        # Select states in which spin i flipped and use it create object for EP estimation 
#        g_samples  = observables.get_g_observables(S, F, i)
#        if len(g_samples) > 0:
#            g_mean     = g_samples.mean(axis=0)
#            sigma_emp += frequencies[i] * get_spin_empirical_EP(beta, J, i, g_mean)

#    return sigma_emp


# def get_spin_empirical_EP(beta, J, i, g_mean):
#    # Calculate ``ground truth'' EP contribution from spin i. Here g_mean is the expectation of samples
#    # return by observables.get_g_observables, i.e., 
#    #       g_mean = observables.get_g_observables(S, F, i).mean(axis=0) 

#    # Calculate contributions to empirical EP from observables of spin i
#    J_i         = utils.torch_to_numpy(J[i,:])
#    J_i_no_i    = np.hstack([J_i[:i], J_i[i+1:]])   # remove i'th entry, due to our convention
#    spin_emp    = float(beta) * J_i_no_i @ utils.torch_to_numpy(g_mean)

#    return spin_emp
