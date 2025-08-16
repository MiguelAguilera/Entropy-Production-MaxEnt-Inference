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

@njit(f'{DTYPE}({DTYPE}[::1], {DTYPE}[::1])', inline='always')
def GlauberStep(betaJi, s):
    """
    Perform a single Glauber update for a given spin.

    Arguments:
        betaJi (array)  : Couplings to other N spins, multiplier by inverse temperature beta
        s (array)       : Current spin configuration.

    Returns:
        int              : New value of the spin (-1 or 1).
    """
    return float(int(np.random.rand() < (np.tanh(betaJi @ s) + 1) / 2) * 2 - 1)


@njit(f'{DTYPE}[:]({DTYPE}[:,::1], {DTYPE}[::1], int32)', inline='always')
def ParallelGlauberStep(betaJ, s, T=1):
    """
    Parallel Glauber dynamics: all spins are updated simultaneously.

    Args:
        J (matrix): Coupling matrix, multiplied by inverse temperature beta.
        s (array): Initial spin state.
        T (int): Number of Monte Carlo sweeps.

    Returns:
        array: Final spin configuration.
    """
    size = len(s)
    for _ in range(T):
        s_p = s.copy()
        for i in range(size):
            s[i] = GlauberStep(betaJ[i, :], s_p)
    return s

@njit(f'{DTYPE}[:]({DTYPE}[:,::1], {DTYPE}[::1], {DTYPE}[::1], int32)', inline='always')
def ParallelGlauberStepAnneal(J, beta_values, s, T=1):
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
            s[i] = GlauberStep(J[i, :] * beta_values[t*size + i], s_p)
    return s


@njit(parallel=True, fastmath=True)
def run_simulation(beta, J, warmup_fraction=0.1, 
                   warmup_steps=100, samples_per_spin=1_000_000, warmup_anneal=True,
                   thinning_multiplier=1,
                   num_restarts=1000, sequential=True, progressbar=True, seed=None):
    """
    Monte Carlo sampling of nonequilibrium spin model using Glauber dynamics.
    By default, we use a thinning factor of N between samples

    Args:
        beta (float)                : Inverse temperature
        J (2d np.array)             : NxN matrix of coupling coefficients
        warmup_fraction (float)     : Fraction of Monte Carlo steps in the warm-up at the beginning of each restart
        warmup_steps (int)          : N*warmup_steps is minimum number of MC steps that will be taken for warmup
        warmup_anneal (bool)        : Whether to use annealing schedule during warmup
        samples_per_spin (int)      : Number of samples per spin to return
        thinning_multiplier (int)   : In between these samples, we reduce correlations by discarding 
                                       thinning_multiplier * number of samples
        num_restarts (int)          : Number of times to restart sampler
        sequential (bool)           : Whether to use sequential or parallel updates
        progressbar (bool)          : Whether to display progressbar during simulation
        seed (int)                  : Random seed for reproducibility

    Returns:
        S: Nxsamples_per_spin int array : Samples of -1,1 state from stationary distribution
        F: Nxsamples_per_spin bool array : Samples of state transitions (True: flipped, False: no flip)
    """

    N = J.shape[0]
    J = J.astype(DTYPE)
    betaJ = (beta*J).astype(DTYPE)

    samples_per_spin = int(samples_per_spin)

    # Define matrix of spin states and spin flips      
    S = np.empty((samples_per_spin, N), dtype=np.int8)
    F = np.empty((samples_per_spin, N), dtype=np.bool_)

    samples_per_restart = samples_per_spin//num_restarts
    assert samples_per_restart > 0, f"num_restarts={num_restarts} is too large (larger than samples_per_spin={samples_per_spin}"

    print_every = int(samples_per_spin/100)
    if progressbar:
        print("-"*100)

    use_warmup_steps = int(max(warmup_fraction * samples_per_restart, warmup_steps))
    if warmup_anneal:
        tail_frac = 0.2
        n_tail = int(N * use_warmup_steps * tail_frac)
        n_ramp = N * use_warmup_steps - n_tail
        ramp = np.linspace(0, beta, n_ramp)
        tail = [beta,]* n_tail
        anneal_betas = np.array(list(ramp) + tail, dtype=DTYPE)
    else:
        anneal_betas = (np.ones(N * use_warmup_steps)*beta).astype(DTYPE)

    for restart_ix in prange(num_restarts):
        if seed is not None:
            np.random.seed(seed + restart_ix*10)

        # Start from a random state, then warm up for N * warmup * samples_per_restart steps
        s = ((np.random.randint(0, 2, N) * 2) - 1).astype(DTYPE)

        if sequential:
            spin_indices = np.random.randint(0, N, N * use_warmup_steps)
            if warmup_anneal:
                for ix, spin in enumerate(spin_indices):
                    s[spin] = GlauberStep(J[spin, :]*anneal_betas[ix], s)
            else:
                for spin in spin_indices:
                    s[spin] = GlauberStep(betaJ[spin, :], s)
        else:
            if warmup_anneal:
                s = ParallelGlauberStepAnneal(J, anneal_betas, s, T=use_warmup_steps)
            else:
                s = ParallelGlauberStep(betaJ, s, T=use_warmup_steps)


        # Now draw samples from steady state
        for r in range(samples_per_restart):
            # First, thin out samples by N steps to decrease correlations
            if sequential:
                indices = np.random.randint(0, N, int(N*thinning_multiplier))
                for i in indices:
                    s[i] = GlauberStep(betaJ[i, :], s)

            else:
                s = ParallelGlauberStep(betaJ, s, T=thinning_multiplier)
            
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


def convert_to_nonmultipartite(S, F):
    # Convert samples S and F to non-multipartite form
    # Arguments:
    #   S: samples_per_spin x N (int)  : initial state ∈ {-1,1}^N at beginning of each of samples_per_spin x N transition attempts
    #   F: samples_per_spin x N (bool) : whether each spin flipped or not in each of samples_per_spin x N transition attempts
    # Returns:
    #   X0: (samples_per_spin x N) x N : initial states for each spin attempt
    #   X1: (samples_per_spin x N) x N : final states for each spin attempt

    # Stack S and F into non-multipartite datastructure
    X0s, X1s = [], []
    N = S.shape[1]
    for i in range(N):
        X0s.append(S)
        Sp = S.copy()
        Sp[F[:,i],i] *= -1
        X1s.append(Sp)
    X0 = np.vstack(X0s)
    X1 = np.vstack(X1s)

    # Shuffle data
    perm = np.random.permutation(X0.shape[0])
    return X0[perm], X1[perm]



# ***** Entropy production estimates from empirical statistics *****

import observables
import utils
def get_empirical_EP(beta, J, S, F):
    # Calculate empirical EP from samples S and F
    num_samples_per_spin, N = S.shape
    total_flips = N * num_samples_per_spin  # Total spin-flip attempts

    # Running sums to keep track of EP estimates
    sigma_emp = 0.0

    # Because system is multipartite, we can separately estimate EP for each spin

    frequencies = F.sum(axis=0)/total_flips      # frequency of spin i flips
    for i in range(N):
        # Select states in which spin i flipped and use it create object for EP estimation 
        g_samples  = observables.get_g_observables(S, F, i)
        if len(g_samples) > 0:
            g_mean     = g_samples.mean(axis=0)
            sigma_emp += frequencies[i] * get_spin_empirical_EP(beta, J, i, g_mean)

    return sigma_emp


def get_spin_empirical_EP(beta, J, i, g_mean):
    # Calculate ``ground truth'' EP contribution from spin i. Here g_mean is the expectation of samples
    # return by observables.get_g_observables, i.e., 
    #       g_mean = observables.get_g_observables(S, F, i).mean(axis=0) 

    # Calculate contributions to empirical EP from observables of spin i
    J_i         = utils.torch_to_numpy(J[i,:])
    J_i_no_i    = np.hstack([J_i[:i], J_i[i+1:]])   # remove i'th entry, due to our convention
    spin_emp    = float(beta) * J_i_no_i @ utils.torch_to_numpy(g_mean)

    return spin_emp
