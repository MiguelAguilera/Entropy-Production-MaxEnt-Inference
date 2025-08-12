#!/usr/bin/env python3
"""

Given a linear Langevin / OU system:
    dx = A x dt + b dt + √2 B dW_t
with D = 2 B B^T, this script computes:
  1) Stationary covariance Sigma_inf (if it exists) solving
         A Σ + Σ A^T + D = 0
  2) Delayed covariance for lag t (can be positive or negative):
         Σ(t,0) = e^{A t} Σ(0,0)        if t >= 0
         Σ(t,0) = Σ(0,0) e^{A^T (-t)}   if t < 0
Note: Stationarity requires Re(eigs(A)) < 0.
"""

import numpy as np
import scipy.linalg as sla


def lyapunov_continuous(A: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Solve A X + X A^T + D = 0 for X.
    Prefers SciPy's solver if available; otherwise uses vec-trick.
    """
    n, m = A.shape
    assert n == m, "A must be square"
    assert D.shape == (n, n), "D must be n x n"

    return sla.solve_continuous_lyapunov(A, -D)


def stationary_covariance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute Σ_inf from A and B, with D = 2 B B^T.
    """
    D = 2 * B @ B.T
    return lyapunov_continuous(A, D)

def ep_gaussian(Sigma00: np.ndarray,
                                    Sigma11: np.ndarray,
                                    Sigma01: np.ndarray) -> float:
    """
    Entropy production (KL forward || time-swapped) for a zero-mean joint Gaussian,
    specified by its block covariances.

    Inputs:
    Sigma00 : Cov[x0]  (n x n)
    Sigma11 : Cov[x1]  (n x n)
    Sigma01 : Cov[x0, x1] (n x n)

    Returns:
    sigma >= 0 (up to numerical error)
    """

    n = Sigma00.shape[0]
    assert Sigma00.shape == (n, n)
    assert Sigma11.shape == (n, n)
    assert Sigma01.shape == (n, n)

    # Build forward and swapped joint covariances
    S = np.block([[Sigma00,         Sigma01       ],
                [Sigma01.T,       Sigma11       ]])
    Srev = np.block([[Sigma11,       Sigma01.T     ],
                    [Sigma01,       Sigma00       ]])

    # Stable solve instead of inverse
    Sinv_rev_S = np.linalg.solve(Srev, S)
    sigma = 0.5 * (np.trace(Sinv_rev_S) - 2*n)

    # Ensure a clean real scalar
    return float(np.real_if_close(sigma))

if __name__ == "__main__":

    N=20
    beta = 1
    t = 1  # lag time
    T=1_000_000    # Number of samples to draw

    print(f"Generating {T} samples for N={N} of an Ornstein-Uhlenbeck process with lag time t={t}")

    # Set random seed for reproducibility
    seed = 4222
    rng = np.random.default_rng(seed)

    A = -np.eye(N) + rng.standard_normal((N, N)) * 0.1# Random matrix
    B = np.eye(N)* beta   # Identity matrix as noise term
    

    # Check eigenvalues
    eigs = np.linalg.eigvals(A)
    if np.max(np.real(eigs)) >= 0:
        raise ValueError("The system is not stable, largest eigenvalue is non-negative.")   
    
    Sigma_inf = stationary_covariance(A, B)
    F = sla.expm(A * t)            # F = e^{At}
    C_t0 = F @ Sigma_inf           # Σ(t,0) = e^{At} Σ

    Q = Sigma_inf - F @ Sigma_inf @ F.T
    
    # Sample x0
    L0 = np.linalg.cholesky(Sigma_inf)
    Z0 = rng.standard_normal((T, N))
    X0 = Z0 @ L0.T

    # Conditional for xt | x0
    LQ = np.linalg.cholesky(Q)
    Zt = rng.standard_normal((T, N))
    Xt = (X0 @ F.T) + Zt @ LQ.T


    sigma = ep_gaussian(Sigma_inf, Sigma_inf, C_t0)
    print("Empirical entropy production (KL forward || time-swapped):", sigma)


    import ep_estimators
    import observables  
    import time

    print()
    print(f"Calculating entropy production estimates")

    g_samples = np.vstack([ Xt[:,i]*X0[:,j] - X0[:,i]*Xt[:,j] 
                            for i in range(N) for j in range(i+1, N) ]).T
    data     = observables.CrossCorrelations1(X0, Xt)
    observable_desc = "Xt[:,i]*X0[:,j] - X0[:,i]*Xt[:,j] for i<j"



    g_mean = g_samples.mean(axis=0)
    data             = observables.Dataset(g_samples=g_samples)

    np.random.seed(42) # Set seed for reproducibility of holdout shuffles
    train, val, test = data.split_train_val_test()

    stime            = time.time()
    sigma_N_obs, _   = ep_estimators.get_EP_Newton1Step(train, validation=val, test=test)
    time_N_obs       = time.time() - stime


    stime = time.time()
    sigma_G_obs, _   = ep_estimators.get_EP_Estimate(train, validation=val, test=test)
    time_G_obs       = time.time() - stime
    



    print(f"Observables gᵢⱼ(x) = {observable_desc}")
    print(f"  Σ_g   (From observable samples, gradient ascent) :    {sigma_G_obs :.6f}  ({time_G_obs    :.3f}s)")
    print(f"  Σ̂_g   (From observable samples, 1-step Newton  ) :    {sigma_N_obs :.6f}  ({time_N_obs    :.3f}s)")