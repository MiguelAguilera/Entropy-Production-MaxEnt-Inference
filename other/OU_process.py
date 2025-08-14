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

import argparse
import os, sys
import matplotlib.pyplot as plt

sys.path.insert(0, '..')

import ep_estimators
import observables  
import time

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

    parser = argparse.ArgumentParser(description="OU EP vs lag time with optional estimators overlay")
    parser.add_argument("--N", type=int, default=10, help="System size")
    parser.add_argument("--beta", type=float, default=1.0, help="Noise scale for B = beta*I")
    parser.add_argument("--seed", type=int, default=4222, help="RNG seed")
    parser.add_argument("--t_min", type=float, default=0.01, help="Minimum lag time")
    parser.add_argument("--t_max", type=float, default=4.0, help="Maximum lag time")
    parser.add_argument("--num_t", type=int, default=21, help="Number of lag times")
    parser.add_argument("--estimate", action="store_true", default=False,
                        help="Overlay EP estimates using ep_estimators/observables")
    parser.add_argument("--T", type=int, default=1_000_000,
                        help="Samples to draw when --estimate is used")
    parser.add_argument("--no_plot", action="store_true", default=False, help="Disable plt.show()")
    parser.add_argument("--NEEP", action="store_true", default=False, help="Use NEEP objective")
    parser.add_argument("--Newton", action="store_true", default=False, help="Run Newtons method")
    parser.add_argument("--partial", action="store_true", default=False, help="Only use subset of antisymmetric observables")

    args = parser.parse_args()

    if args.NEEP:
        print("Using NEEP")
        from NEEP import DatasetNEEP
        dataset_class = DatasetNEEP
    else:
        dataset_class = observables.Dataset



    # RNG and lags
    rng = np.random.default_rng(args.seed)
    t_values = np.linspace(args.t_min, args.t_max, args.num_t)
    Num_t = len(t_values)


    N=args.N
    beta = args.beta
    T=args.T

    # Set random seed for reproducibility
    seed = args.seed
    rng = np.random.default_rng(seed)

    A = -np.eye(N) + rng.standard_normal((N, N)) * 0.2 # Random matrix
    B = np.eye(N)* beta   # Identity matrix as noise term
    

    # Check eigenvalues
    eigs = np.linalg.eigvals(A)
    if np.max(np.real(eigs)) >= 0:
        raise ValueError("The system is not stable, largest eigenvalue is non-negative.")   


    Sigma_inf = stationary_covariance(A, B)

    sigma_emp = np.zeros(Num_t)
    sigma_g = np.zeros(Num_t)
    sigma_hat_g = np.zeros(Num_t)
    sigma_MTUR_g = np.zeros(Num_t)

    for i, t in enumerate(t_values):
        print()
        print(f"Generating {T} samples for N={N} of an Ornstein-Uhlenbeck process with lag time t={t}")
        
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




        print(f"Calculating entropy production estimates")

        # g_samples = np.vstack([ Xt[:,i]*X0[:,j] - X0[:,i]*Xt[:,j] 
        #                         for i in range(N) for j in range(i, N) ]).T
        # data     = observables.CrossCorrelations1(X0, Xt)
        # observable_desc = "Xt[:,i]*X0[:,j] - X0[:,i]*Xt[:,j] for i<j"

        g_samples_0t = np.vstack([ X0[:,i]*Xt[:,j]
                                for i in range(N) for j in range(N) ]).T
        g_samples_t0 = np.vstack([ Xt[:,i]*X0[:,j]
                                for i in range(N) for j in range(N) ]).T
        # g_samples  = np.hstack([g_samples_0, g_samples_0t, g_samples_t])
        # rev_g_samples = np.hstack([g_samples_t, g_samples_t0, g_samples_0])
        # data             = observables.Dataset(g_samples=g_samples, rev_g_samples=rev_g_samples)

        if args.partial:
            g_samples  = np.hstack([g_samples_0t-g_samples_t0])
            observable_desc = "<x_i(0)x_j(t)-x_i(t)x_j(0)>"
        else:
            g_samples_0 = np.vstack([ X0[:,i]*X0[:,j]
                                    for i in range(N) for j in range(i,N) ]).T
            g_samples_t = np.vstack([ Xt[:,i]*Xt[:,j]
                                    for i in range(N) for j in range(i,N) ]).T
            g_samples  = np.hstack([g_samples_t-g_samples_0, g_samples_0t-g_samples_t0])
            observable_desc = "<x_i(0)x_j(t)-x_i(t)x_j(0)>,<x_i(0)x_j(0)-x_i(t)x_j(t)>"


        data             = dataset_class(g_samples=g_samples)


        Sigma00 = np.cov(X0.T, bias=False)
        Sigma11 = np.cov(Xt.T, bias=False)
        Sigma01 = (X0.T @ Xt) / (T - 1)

        sigma_emp[i] = ep_gaussian(Sigma00, Sigma11, Sigma01)

        np.random.seed(42) # Set seed for reproducibility of holdout shuffles
        train, val, test = data.split_train_val_test()

        if args.Newton:
            stime            = time.time()
            sigma_N_obs, _   = ep_estimators.get_EP_Newton1Step(train, validation=val, test=test)
            time_N_obs       = time.time() - stime
            sigma_hat_g[i] = sigma_N_obs   

        stime = time.time()
        sigma_G_obs, _   = ep_estimators.get_EP_Estimate(train, validation=val, test=test)
        time_G_obs       = time.time() - stime
        sigma_g[i] = sigma_G_obs

        stime = time.time()
        sigma_MTUR_obs, _ = ep_estimators.get_EP_MTUR(data)
        time_MTUR_obs     = time.time() - stime
        sigma_MTUR_g[i] = sigma_MTUR_obs 

        print(f"Observables gᵢⱼ(x) = {observable_desc}")
        print(f"  Σ_g   (gradient ascent) :    {sigma_G_obs :.6f}  ({time_G_obs    :.3f}s)")
        if args.Newton:
            print(f"  Σ̂_g   (1-step Newton  ) :    {sigma_N_obs :.6f}  ({time_N_obs    :.3f}s)")
        print(f"  Σ̂_g   (MTUR) :    {sigma_MTUR_obs :.6f}  ({time_MTUR_obs    :.3f}s)")

        

    # -------------------------------
    # Plot Results
    # -------------------------------
    plt.rc('text', usetex=True)
    plt.rc('font', size=22, family='serif', serif=['latin modern roman'])
    plt.rc('legend', fontsize=20)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, sigma_emp / t_values, label='Empirical EP', marker='o')
    plt.plot(t_values, sigma_g / t_values, label='EP Estimate (Gradient Ascent)', marker='x')
    if args.Newton:
        plt.plot(t_values, sigma_hat_g / t_values, label='EP Estimate (1-step Newton)', marker='s')
    plt.plot(t_values, sigma_MTUR_g / t_values, label='EP Estimate (MTUR)', marker='d')
    plt.xlabel(r'Lag Time $t$')
    plt.ylabel(r'EP / $t$')
    plt.legend()
    plt.tight_layout()  
    plt.show() if not args.no_plot else None