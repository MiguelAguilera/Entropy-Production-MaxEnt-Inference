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

import os, sys, time
import argparse
import matplotlib.pyplot as plt

sys.path.insert(0, '..')

import ep_estimators
import observables  
import NEEP

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

    parser = argparse.ArgumentParser(description="OU EP vs lag time with optional estimators overlay",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--N", type=int, default=10, help="System size")
    parser.add_argument("--seed", type=int, default=4222, help="RNG seed (-1 for no seed)")
    parser.add_argument("--t_min", type=float, default=0.01, help="Minimum lag time")
    parser.add_argument("--t_max", type=float, default=4, help="Maximum lag time")
    parser.add_argument("--num_t", type=int, default=51, help="Number of lag times")
    #parser.add_argument("--estimate", action="store_true", default=False,
    #                    help="Overlay EP estimates using ep_estimators/observables")
    parser.add_argument("--T", type=int, default=4_000_000, help="Samples to draw")
    parser.add_argument("--no_plot", action="store_true", default=False, help="Disable plt.show()")
    parser.add_argument("--NEEP", action="store_true", default=False, help="Use NEEP objective")
    parser.add_argument("--Newton", action="store_true", default=False, help="Run Newtons method")
    parser.add_argument("--partial", action="store_true", default=False, help="Only use subset of antisymmetric observables")

    args = parser.parse_args()

    # if args.NEEP:
    #     print("Using NEEP")
    #     from NEEP import DatasetNEEP
    #     dataset_class = DatasetNEEP
    # else:
    #     dataset_class = observables.Dataset



    t_values = np.linspace(args.t_min, args.t_max, args.num_t)
    t_values = np.logspace(np.log10(args.t_min), np.log10(args.t_max), args.num_t)
    Num_t = len(t_values)


    N=args.N
    T=args.T

    # RNG and lags
    # Set random seed for reproducibility
    if args.seed != -1:
        seed = args.seed
        rng = np.random.default_rng(seed)
        np.random.seed(args.seed)
    else:
        rng = np.random.default_rng()

    off_diag = rng.standard_normal((N, N)) * 0.2 # Random matrix
    if False:
        off_diag = np.sign(off_diag)*0.25
        #off_diag[np.random.random((N,N))>.1]=0
        np.fill_diagonal(off_diag, 0)  # Zero diagonal
    A = -np.eye(N) + off_diag 
    B = np.eye(N)   # Identity matrix as noise term
    

    # Check eigenvalues
    eigs = np.linalg.eigvals(A)
    if np.max(np.real(eigs)) >= 0:
        raise ValueError("The system is not stable, largest eigenvalue is non-negative.")   


    Sigma_inf = stationary_covariance(A, B)

    sigma_emp = np.zeros(Num_t)
    sigma_g = np.zeros(Num_t)
    sigma_g_neep = np.zeros(Num_t)
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
#        data     = CrossCorrelations3(X0, Xt)
#        data     = observables.CrossCorrelations1(X0, Xt)
        
        
        
        observable_desc = "Xt[:,i]*X0[:,j] - X0[:,i]*Xt[:,j] for i<j"

        g_samples_cross = np.vstack([ Xt[:,i]*X0[:,j] - X0[:,i]*Xt[:,j]
                                for i in range(N) for j in range(i,N) ]).T
        g_samples_same = np.vstack([ Xt[:,i]*Xt[:,j] - X0[:,i]*X0[:,j]
                                for i in range(N) for j in range(i,N) ]).T
        if args.partial:
            g_samples  = g_samples_cross
            observable_desc = "<x_i(0)x_j(t)-x_i(t)x_j(0)>"
        else:
            g_samples  = np.hstack([g_samples_cross, g_samples_same])
            observable_desc = "<x_i(0)x_j(t)-x_i(t)x_j(0)>,<x_i(0)x_j(0)-x_i(t)x_j(t)>"
        del g_samples_cross, g_samples_same

#        g_samples = np.tanh(g_samples)*g_samples**2
        data             = observables.Dataset(g_samples=g_samples)
        dataN            = NEEP.DatasetNEEP(g_samples=g_samples)

        Sigma00 = np.cov(X0.T, bias=False)
        Sigma11 = np.cov(Xt.T, bias=False)
        Sigma01 = (X0.T @ Xt) / (T - 1)

        sigma_emp[i] = ep_gaussian(Sigma00, Sigma11, Sigma01)

        np.random.seed(42) # Set seed for reproducibility of holdout shuffles
        train , val , test  = data.split_train_val_test()
        if args.NEEP:
            trainN, valN, testN = dataN.split_train_val_test()

        if args.Newton:
            stime            = time.time()
            sigma_N_obs, _   = ep_estimators.get_EP_Newton1Step(train, validation=val, test=test)
            time_N_obs       = time.time() - stime
            sigma_hat_g[i] = sigma_N_obs   

        stime = time.time()
        sigma_G_obs, _   = ep_estimators.get_EP_Estimate(train, validation=val, test=test)
        time_G_obs       = time.time() - stime
        sigma_g[i] = sigma_G_obs

#        if args.NEEP:
#            stime = time.time()
#            sigma_G_NEEP_obs, _   = ep_estimators.get_EP_Estimate(trainN, validation=valN, test=testN, verbose=True, optimizer_kwargs={'tol':1e-30})
#            time_G_NEEP_obs       = time.time() - stime
#            sigma_g_neep[i] = sigma_G_NEEP_obs

        stime = time.time()
        sigma_MTUR_obs, _ = ep_estimators.get_EP_MTUR(data)
        time_MTUR_obs     = time.time() - stime
        sigma_MTUR_g[i] = sigma_MTUR_obs 

        print(f"Observables gᵢⱼ(x) = {observable_desc}")
        print(f"  Σ_g   (gradient ascent) :    {sigma_G_obs :.6f}  ({time_G_obs    :.3f}s)")
        if args.NEEP:
            print(f"  Σ_g   (grad. asc. NEEP) :    {sigma_G_NEEP_obs :.6f}  ({time_G_NEEP_obs    :.3f}s)")
        if args.Newton:
            print(f"  Σ̂_g   (1-step Newton  ) :    {sigma_N_obs :.6f}  ({time_N_obs    :.3f}s)")
        print(f"  Σ̂_g   (MTUR           ) :    {sigma_MTUR_obs :.6f}  ({time_MTUR_obs    :.3f}s)")

        
    if not args.no_plot:
        # -------------------------------
        # Plot Results (styled like spin EP plots)
        # -------------------------------
        plt.rc('text', usetex=True)
        plt.rc('font', size=22, family='serif', serif=['latin modern roman'])
        plt.rc('legend', fontsize=20)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')

        labels = [
            r'$\Sigma$',                        # Empirical EP
            r'${\Sigma}_{\bm g}$',              # Gradient ascent
            r'$\widehat{\Sigma}_{\bm g}$',      # Newton-1
            r'$\Sigma_{\bm g}^\textnormal{\small TUR}$'  # MTUR
        ]

        cmap = plt.get_cmap('inferno_r')
        colors = [cmap(0.25), cmap(0.5), cmap(0.75)]

        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot empirical EP first (thick dashed black)
        ax.plot(t_values, sigma_emp / t_values,
                'k', linestyle=(0, (2, 3)), label=labels[0], lw=3)

        # Plot estimators
        ax.plot(t_values, sigma_g / t_values, label=labels[1], color=colors[2], lw=2)
        if args.Newton:
            ax.plot(t_values, sigma_hat_g / t_values, label=labels[2], color=colors[1], lw=2)
        ax.plot(t_values, sigma_MTUR_g / t_values, label=labels[3], color=colors[0], lw=2)

        # Replot empirical for emphasis
        ax.plot(t_values, sigma_emp / t_values, 'k', linestyle=(0, (2, 3)), lw=3)

        # Axes & labels
#        ax.set_xscale('log')  # since times are logarithmic now
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'EP / $t$', labelpad=20)
        ax.set_xlim([t_values[0], t_values[-1]])
        ax.set_ylim([0, 1.05 * max(np.max(sigma_emp / t_values),
                                   np.max(sigma_g / t_values),
                                   np.max(sigma_hat_g / t_values) if args.Newton else 0,
                                   np.max(sigma_MTUR_g / t_values))])

        # Legend
        legend = ax.legend(
            ncol=1,
            columnspacing=0.25,
            handlelength=1.0,
            handletextpad=0.25,
            labelspacing=0.25,
            loc='best'
        )

        plt.tight_layout()
        # -------------------------------
        # Save Plot
        # -------------------------------
        IMG_DIR = 'img'
        if not os.path.exists(IMG_DIR):
            print(f'Creating image directory: {IMG_DIR}')
            os.makedirs(IMG_DIR)

        plt.savefig(f'{IMG_DIR}/Fig_OU.pdf', bbox_inches='tight', pad_inches=0.1)

        # Show plot (unless disabled)
        if not args.no_plot:
            plt.show()
