import numpy as np
from numba import njit, prange
import h5py
import hdf5plugin
import threading
import time

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
        F[:, r] = (- s1 * s).astype('int32') // 2  # Indicates if spin changed: 1 if flipped, -1 otherwise

    return S, F



# --------- Simulation Orchestration Functions --------- #

def save_data(file_name, J, H, S, F):
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('J', data=J, compression='gzip', compression_opts=5)
        f.create_dataset('H', data=H, compression='gzip', compression_opts=5)

    for i in range(F.shape[0]):
        idxs = np.where(F[i, :] == 1)[0]
        S_i = S[:, idxs] if len(idxs) > 0 else np.zeros((S.shape[0], 0), dtype=bool)
        with h5py.File(file_name, 'a') as f:
#            f.create_dataset(f'S_{i}', data=((S_i + 1) // 2).astype(bool), compression='gzip', compression_opts=5)
            bool_array = ((S_i + 1) // 2).astype(bool)
            f.create_dataset(
                f'S_{i}',
                data=bool_array,
                **hdf5plugin.Blosc(cname='zstd', clevel=4, shuffle=hdf5plugin.Blosc.BITSHUFFLE)
            )
    print(f"[Thread] Data saved to {file_name}")
    
def run_simulation(file_name, N, num_steps=128, rep=1_000_000,
                   beta=1.3485, J0=1.0, DJ=0.5, seed=None,
                   onlychanges=None, sequential=True):
    """
    Run Glauber dynamics simulation and save results.

    Args:
        file_name (str): Output file name (HDF5).
        N (int): Number of spins.
        num_steps (int): Number of Glauber steps per sample.
        rep (int): Number of repetitions/samples.
        beta (float): Inverse temperature.
        J0 (float): Mean coupling.
        DJ (float): Coupling variability.
        seed (int): Random seed.
        onlychanges (None or bool): Not used.
        sequential (bool): Whether to use sequential or parallel updates.
    """

    if seed is not None:
        np.random.seed(seed)

    start_time = time.time()

    # Initialize couplings and fields
    rnd = np.random.randn(N, N)
    J = beta * (J0 / N + rnd.astype(DTYPE) * DJ / np.sqrt(N))
    np.fill_diagonal(J, 0)
    H = np.zeros(N, dtype=DTYPE)

    # Warm-up run (just-in-time compilation trigger)
    _ = sample(1, H, J, 1)

    # Actual sampling
    S, F = sample(rep, H, J, num_steps, sequential=sequential)

    print('Sampled states: %d' % S.shape[1])
    print('   - state changes : %d' % (F==1).sum())
    print('   - magnetization : %f' % np.mean(S.astype(float)))

    # Save model parameters
    t = threading.Thread(target=save_data, args=(file_name, J, H, S, F))
    t.start()
    print(f"Saving data. Took {time.time()-start_time:.3f}s. Main loop continues...")
#    with h5py.File(file_name, 'w') as f:
#        f.create_dataset('J', data=J, compression='gzip', compression_opts=5)
#        f.create_dataset('H', data=H, compression='gzip', compression_opts=5)

#    # Save spin configurations that led to a flip
#    for i in range(N):
#        idxs = np.where(F[i, :] == 1)[0]
#        if len(idxs) == 0:
#            with h5py.File(file_name, 'a') as f:
#                f.create_dataset(f'S_{i}', data=np.zeros((N, 0), dtype=bool), compression='gzip', compression_opts=5)
#            continue

#        S_i = S[:, idxs]
#        with h5py.File(file_name, 'a') as f:
#            f.create_dataset(f'S_{i}', data=((S_i + 1) // 2).astype(bool), compression='gzip', compression_opts=5)

#    print(f"Data saved to {file_name}, time = %.3f seconds" % (time.time() - start_time))

