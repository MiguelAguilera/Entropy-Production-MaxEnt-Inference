# Nonequilibrium spin model used in the manuscript:
#  Aguilera, Ito, Kolchinsky, Inferring entropy production in many-body 
#                             systems using nonequilibrium MaxEnt
#
# We include code to generate random coupling matrices and to perform Monte Carlo 
# simulations using Glauber dynamics. Here we use the convention that the coupling 
# matrix is indicated as J (not w as in the manuscript)

import numpy as np
from numba import njit, objmode

DTYPE = 'float32'  # Default data type for numerical operations

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


def get_couplings_patterns(N, L):
    """
    Generate coupling matrix for nonequilibrium spin model with pattern sequence.
    
    See for example: A Düring et al 1998 J. Phys. A: Math. Gen. 31 8607

    Args:
        N (int)    : Number of spins
        L (int)    : Number of random patterns to store
    """
    # Generate random patterns of -1 and 1s
    xi = np.random.randint(0,2, (L,N) ).astype(DTYPE)*2-1
    xi_shifted = np.zeros_like(xi)
    for a in range(L-1):
        xi_shifted[a+1,]= xi[a,:]
    xi_shifted[0,:]= -xi[-1,:]
    J = (xi.T @ xi_shifted / N).astype(DTYPE)
    np.fill_diagonal(J, 0)
    return J



# ***** Monte Carlo code  *****

@njit('float32(float32[::1], float32[::1])', inline='always')
def GlauberStep(Ji, s):
    """
    Perform a single Glauber update for a given spin.

    Arguments:
        Ji (array)  : Couplings to other N spins
        s (array)   : Current spin configuration.

    Returns:
        int         : New value of the spin (-1 or 1).
    """
    return float(int(np.random.rand() < (np.tanh(Ji @ s) + 1) / 2) * 2 - 1)


@njit('float32[:](float32[:,::1], float32[::1], int32)', inline='always')
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
            s[i] = GlauberStep(J[i, :], s_p)
    return s


@njit(parallel=True, fastmath=True, cache=True)
def run_simulation(beta, J, warmup=0.1, samples_per_spin=1_000_000, 
                   num_restarts=1, sequential=True, progressbar=True):
    """
    Monte Carlo sampling of nonequilibrium spin model using Glauber dynamics.
    By default, we use a thinning factor of N between samples

    Args:
        beta (float)                : Inverse temperature
        J (2d np.array)             : NxN matrix of coupling coefficients
        warmup (float)              : Fraction of Monte Carlo steps in the warm-up at the beginning of each restart
        samples_per_spin (int)      : Number of samples per spin to return
                                      In between these samples, we use a thinning factor of N
                                      throwaway samples
        num_restarts (int)          : Number of times to restart sampler
        sequential (bool)           : Whether to use sequential or parallel updates
        progressbar (bool)          : Whether to display progressbar during simulation

    Returns:
        S: Nxsamples_per_spin int array : Samples of -1,1 state from stationary distribution
        F: Nxsamples_per_spin bool array : Samples of state transitions (True: flipped, False: no flip)
    """
    N = J.shape[0]
    betaJ = (beta*J).astype(DTYPE)

    # Define matrix of spin states and spin flips      
    S = np.empty((samples_per_spin, N), dtype=np.int8)
    F = np.empty((samples_per_spin, N), dtype=np.bool_)

    samples_per_restart = samples_per_spin//num_restarts
    
    print_every = int(samples_per_spin/100)
    if progressbar:
        print("-"*100)

    for restart_ix in range(num_restarts):
        # Start from a random state, then warm up for N * warmup * samples_per_restart steps
        s = ((np.random.randint(0, 2, N) * 2) - 1).astype(DTYPE)
        if sequential:
            indices = np.random.randint(0, N, int(N * warmup * samples_per_restart))
            for i in indices:
                s[i] = GlauberStep(betaJ[i, :], s)
        else:
            s = ParallelGlauberStep(betaJ, s, T=int(samples_per_restart * warmup))

        # Now draw samples from steady state
        for r in range(samples_per_restart):
            # First, thin out samples by N steps to decrease correlations
            if sequential:
                indices = np.random.randint(0, N, int(N))
                for i in indices:
                    s[i] = GlauberStep(betaJ[i, :], s)

            else:
                s = ParallelGlauberStep(betaJ, s, T=1)
            
            # Save the sampled state
            sample_ix = restart_ix * samples_per_restart + r
            S[sample_ix, :] = s.astype(np.int8)

            # Sample possible flips for each spoin
            s1 = np.empty(N, dtype=DTYPE)
            for i in range(N):
                s1[i] = GlauberStep(betaJ[i, :], s)

            # Indicates if spin changed: 1 if flipped, 0 otherwise
            F[sample_ix, :] = ((1-s1 * s)/2).astype(np.bool_)  

            if progressbar and sample_ix % print_every == 0:
                with objmode():
                    print(".", end="", flush=True)

    if progressbar:
        print()

    return S, F


def get_g_observables(S, F, i):
    # Use S and F to select states in which spin i flipped
    S_i = S[F[:,i],:]

    # Given states S_i in which spin i flipped, calculate observables 
    #        g(x) = g_{ij} = (x_i' - x_i) x_j 
    # where x_i' and x_i indicate the state of spin i after and before the jump, 
    # and x_j is the state of every other spin .
    g_samples = -2 * np.einsum('i,ij->ij', S_i[:, i], S_i)
    
    # We remove the i-th observable because its always -2
    g_samples = np.hstack([g_samples [:,:i], g_samples [:,i+1:]])

    return g_samples


def get_spin_empirical_EP(beta, J, i, g_mean):
    # Calculate ``ground truth'' EP contribution from spin i. Here g_mean is the expectation of samples
    # return by get_g_observables, i.e., 
    #       g_mean = get_g_observables(S, F, i).mean(axis=0) 
    import utils
    # Calculate contributions to empirical EP from observables of spin i
    g_mean_t   = utils.numpy_to_torch(g_mean)
    J_i_t      = utils.numpy_to_torch(J[i,:])
    J_i_t_no_i = utils.remove_i(J_i_t, i)   # remove i'th entry, due to our convention
    return float(beta) * float(J_i_t_no_i @ g_mean_t)
