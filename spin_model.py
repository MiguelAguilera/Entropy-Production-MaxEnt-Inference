import numpy as np
from numba import njit, prange, objmode
import h5py
import hdf5plugin
import threading

DTYPE = 'float32'  # Default data type for numerical operations

# --------- Glauber Dynamics Core Functions --------- #

@njit('float32(float32, float32[::1], float32[::1])', inline='always')
def GlauberStep(Hi, Ji, s):
    """
    Perform a single Glauber update for a given spin.

    Args:
        Hi (float32): Local field.
        Ji (array): Couplings to other spins.
        s (array): Current spin configuration.

    Returns:
        int: New value of the spin (-1 or 1).
    """

    h = Hi + Ji @ s
    return float(int(np.random.rand() < (np.tanh(h) + 1) / 2) * 2 - 1)



@njit('float32[:](float32[::1], float32[:,::1], float32[::1], float32)', inline='always')
def SequentialGlauberStep(H, J, s, T=1):
    """
    Sequential Glauber dynamics: spins are updated using a random order.

    Args:
        H (array): Local fields.
        J (matrix): Coupling matrix.
        s (array): Initial spin state.
        T (int): Number of Monte Carlo sweeps.

    Returns:
        array: Final spin configuration.
    """
    size = len(s)
    indices = np.random.randint(0, size, int(size * T))
    for i in indices:
        s[i] = GlauberStep(H[i], J[i, :], s)
    return s


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


# --------- Sampling Function --------- #

@njit(parallel=True, fastmath=True, cache=True)
def sample(rep, H, J, num_steps, sequential=True,init=0,trials=1000, progressbar=True):
    """
    Sample spin configurations using Glauber dynamics.

    Args:
        rep (int): Number of repetitions/samples.
        H (array): Local fields.
        J (matrix): Coupling matrix.
        num_steps (int): Number of Glauber steps.
        sequential (bool): Whether to use sequential or parallel updates.

    Returns:
        S (array): Matrix of sampled spin configurations.
        F (array): Matrix of spin-flip indicators.
    """
    N = len(H)
    S = np.ones((N, rep), dtype='int32')
    F = np.ones_like(S)

    trial_rep = rep//trials
    
    print_every = int(rep/100)
    if progressbar:
        print("-"*100)

    for trial in range(trials):
        if init==1:
            s0  = np.ones(N, dtype=DTYPE)
        elif init==0:
            s0  = ((np.random.randint(0, 2, N) * 2) - 1).astype(DTYPE)
        else:
            raise Exception('unknown init')

        if sequential:
            s = SequentialGlauberStep(H, J, s0, T=num_steps)
        else:
            s = ParallelGlauberStep(H, J, s0, T=num_steps)
        for r in range(trial_rep):
            if sequential:
                s = SequentialGlauberStep(H, J, s.copy(), T=1)
            else:
                s = ParallelGlauberStep(H, J, s.copy(), T=1)
            
            s1 = np.ones(N, dtype=DTYPE)
            for i in range(N):
                s1[i] = GlauberStep(H[i], J[i, :], s)

            S[:, trial*trial_rep + r] = s.astype('int32')
            F[:, trial*trial_rep + r] = -(s1 * s).astype('int32')  # Indicates if spin changed: 1 if flipped, -1 otherwise

            if progressbar and (trial*trial_rep + r) % print_every == 0:
                with objmode():
                    print(".", end="", flush=True)

    if progressbar:
        print()
    return S, F



def run_simulation(N, num_steps=128, rep=1_000_000, trials=1,
                   beta=1.3485, J0=1.0, DJ=0.5, seed=None,
                   onlychanges=None, sequential=True, 
                   patterns=None):
    """
    Run Glauber dynamics simulation and save results.

    Args:
        N (int): Number of spins.
        num_steps (int): Number of Glauber steps per sample.
        rep (int): Number of repetitions/samples.
        trials (int): How many restarts
        beta (float): Inverse temperature.
        J0 (float): Mean coupling.
        DJ (float): Coupling variability.
        seed (int): Random seed.
        onlychanges (None or bool): Not used.
        sequential (bool): Whether to use sequential or parallel updates.

    Returns:
        J: NxN np.array of coupling coefficients
        H: N-vector of local fields
        S: Nx[rep] np.array of -1/1 initial spin states
        F: Nx[rep] np.array of flips (1) or no flips (0)
    """

    if seed is not None and seed >= 0:
        np.random.seed(seed)

    # Initialize couplings and fields
    rnd = np.random.randn(N, N)
    if patterns is None:
        init=0
        J = (beta * (J0 / N + rnd * DJ / np.sqrt(N))).astype(DTYPE)
    else:
        init=0
        L=patterns
        xi = np.random.randint(0,2,(L,N)).astype(DTYPE)*2-1
        xi_shifted = np.zeros_like(xi)
        for a in range(L-1):
            xi_shifted[a+1,]= xi[a,:]
        xi_shifted[0,:]= -xi[-1,:]
        J = (beta * xi.T @ xi_shifted /N).astype(DTYPE)
    np.fill_diagonal(J, 0)
    H = np.zeros(N, dtype=DTYPE)
    # Warm-up run (just-in-time compilation trigger)
    _ = sample(10, H, J, 1, sequential=sequential,init=init,trials=1,progressbar=False)

    # Actual sampling
    S, F = sample(rep, H, J, num_steps, sequential=sequential,init=init,trials=trials)

    return J, H, S, F
