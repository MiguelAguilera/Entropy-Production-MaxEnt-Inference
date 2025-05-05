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

@njit('float32(float32, float32[::1], float32[::1])', inline='always')
def GlauberStep(Hi, Ji, s):
    """
    Perform a single Glauber update for a given spin.

    Arguments:
        Hi (float32): Local field.
        Ji (array)  : Couplings to other N spins
        s (array)   : Current spin configuration.

    Returns:
        int         : New value of the spin (-1 or 1).
    """
    h = Hi + Ji @ s
    return float(int(np.random.rand() < (np.tanh(h) + 1) / 2) * 2 - 1)


@njit('float32[:](float32[::1], float32[:,::1], float32[::1], int32)', inline='always')
def ParallelGlauberStep(H, J, s, T=1):
    """
    Parallel Glauber dynamics: all spins are updated simultaneously.

    Args:
        H (array): Local fields.
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
            s[i] = GlauberStep(H[i], J[i, :], s_p)
    return s


@njit(parallel=True, fastmath=True, cache=True)
def run_simulation(beta, J, H=None, warmup_steps_per_spin=128, samples_per_spin=1_000_000, 
                   num_restarts=1, sequential=True, progressbar=True):
    """
    Monte Carlo sampling of nonequilibrium spin model using Glauber dynamics.
    By default, we use a thinning factor of N between samples

    Args:
        beta (float)                : Inverse temperature
        J (2d np.array)             : NxN matrix of coupling coefficients
        h (1d np.array)             : N-long vector of local fields (set to all 0s if None)
        warmup_steps_per_spin (int) : Number of Monte Carlo steps in-between saved samples
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
    if H is None:
        H = np.zeros(N, dtype=DTYPE)
    
    betaJ = (beta*J).astype(DTYPE)
    betaH = (beta*H).astype(DTYPE)

    # Explain what happens here        
    S = np.empty((samples_per_spin, N), dtype=np.int8)
    F = np.empty((samples_per_spin, N), dtype=np.bool)

    samples_per_trial = samples_per_spin//num_restarts
    
    print_every = int(samples_per_spin/100)
    if progressbar:
        print("-"*100)

    for trial in range(num_restarts):
        # Start from a random state, then warm up for N * warmup_steps_per_spin steps
        s = ((np.random.randint(0, 2, N) * 2) - 1).astype(DTYPE)
        if sequential:
            indices = np.random.randint(0, N, int(N * warmup_steps_per_spin))
            for i in indices:
                s[i] = GlauberStep(betaH[i], betaJ[i, :], s)
        else:
            s = ParallelGlauberStep(betaH, betaJ, s, T=warmup_steps_per_spin)

        # Now draw samples from steady state
        for r in range(samples_per_trial):
            # First, thin out samples by N steps to decrease correlations
            if sequential:
                indices = np.random.randint(0, N, int(N))
                for i in indices:
                    s[i] = GlauberStep(betaH[i], betaJ[i, :], s)

            else:
                s = ParallelGlauberStep(betaH, betaJ, s, T=1)
            
            # Save the sampled state
            sample_ix = trial*samples_per_trial + r
            S[sample_ix, :] = s.astype(np.int8)

            # Sample possible flips for each spoin
            s1 = np.empty(N, dtype=DTYPE)
            for i in range(N):
                s1[i] = GlauberStep(betaH[i], betaJ[i, :], s)

            # Indicates if spin changed: 1 if flipped, 0 otherwise
            F[sample_ix, :] = ((1-s1 * s)/2).astype(np.bool)  

            if progressbar and sample_ix % print_every == 0:
                with objmode():
                    print(".", end="", flush=True)

    if progressbar:
        print()

    return S, F
