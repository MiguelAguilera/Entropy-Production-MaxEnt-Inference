import numpy as np
from numba import njit, prange
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
    return float(int(np.random.rand() * 2 - 1 < np.tanh(h)) * 2 - 1)



@njit('float32[:](float32[::1], float32[:,::1], float32[::1], int32)', inline='always')
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
    indices = np.random.randint(0, size, size * T)
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
def sample(rep, H, J, num_steps, sequential=True):
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

    for r in prange(rep):
        s   = np.ones(N, dtype='float32')

        if sequential:
            out = SequentialGlauberStep(H, J, s, T=num_steps)
        else:
            out = ParallelGlauberStep(H, J, s, T=num_steps)

        s1 = np.ones(N, dtype='float32')
        for i in range(N):
            s1[i] = GlauberStep(H[i], J[i, :], out)

        S[:, r] = out.astype('int32')
        F[:, r] = -(s1 * s).astype('int32')  # Indicates if spin changed: 1 if flipped, -1 otherwise


    return S, F



def run_simulation(N, num_steps=128, rep=1_000_000,
                   beta=1.3485, J0=1.0, DJ=0.5, seed=None,
                   onlychanges=None, sequential=True):
    """
    Run Glauber dynamics simulation and save results.

    Args:
        N (int): Number of spins.
        num_steps (int): Number of Glauber steps per sample.
        rep (int): Number of repetitions/samples.
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
    J = beta * (J0 / N + rnd.astype(DTYPE) * DJ / np.sqrt(N))
    np.fill_diagonal(J, 0)
    H = np.zeros(N, dtype=DTYPE)

    # Warm-up run (just-in-time compilation trigger)
    _ = sample(1, H, J, 1)

    # Actual sampling
    S, F = sample(rep, H, J, num_steps, sequential=sequential)

    return J, H, S, F
