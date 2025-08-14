#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt


def objective_theta(f, theta, mu, sigma, Nint=4096, zmax=8.0):
    """
    Compute U(theta) = E[ f(theta * X, theta) ] - log E[ exp( -f(theta * X, theta) ) ],
    with X ~ N(mu, sigma^2). Vectorized over theta.

    Parameters
    ----------
    f     : callable, f(x, theta) -> array_like
            Must be NumPy-vectorized over x and theta. When x has shape (N,T)
            and theta has shape (1,T) or (N,T), it should broadcast and return (N,T).
    theta : float or (T,) array
    mu    : float
    sigma : float >= 0
    Nint  : int, number of z grid points
    zmax  : float, half-width for z grid (≈6–8 captures essentially all mass)

    Returns
    -------
    U : (T,) array (or scalar if theta is scalar)
    """
    theta = np.atleast_1d(theta)  # (T,)

    # Degenerate case: X = mu a.s.
    if sigma == 0.0:
        fx = np.asarray(f(theta * mu, theta))    # (T,)
        U  = 2.0 * fx
        return U if U.ndim else U.item()

    # Unit-Gaussian grid: X = mu + sigma z
    z  = np.linspace(-zmax, zmax, Nint)          # (N,)
    dz = (z[-1] - z[0]) / (Nint - 1)
    x  = mu + sigma * z                          # (N,)

    # Standard normal pdf and trapezoid weights -> effective positive weights
    phi = np.exp(-0.5 * z**2) / np.sqrt(2*np.pi) # (N,)
    w   = np.ones_like(z); w[0] = w[-1] = 0.5    # trapezoid end weights
    eff = dz * phi * w                            # (N,)

    # Evaluate f(theta * X, theta) for all theta at once
    X  = x[:, None]                               # (N,1)
    TH = theta[None, :]                           # (1,T)
    fvals = np.asarray(f(X, TH))             # (N,T) via broadcasting
    neg_fvals = np.asarray(f(-X, TH))             # (N,T) via broadcasting

    # First term: E[f(theta X, theta)] over z
    Ef = np.sum(eff[:, None] * fvals, axis=0)     # (T,)

    # Second term: log E[exp(-f(theta X, theta))] via log-sum-exp along z
    m = np.max(neg_fvals, axis=0, keepdims=True)          # (1,T) pivot for stability
    logEexp = np.log(np.sum(eff[:, None] * np.exp(neg_fvals - m), axis=0)) + m.squeeze(0)  # (T,)

    U = Ef - logEexp
    return U if U.ndim else U.item()

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=1.0, help="Mean μ (default=1)")
    ap.add_argument("--sigma", type=float, default=2.0, help="Std dev σ (>0, default=2)")
    ap.add_argument("--theta-min", type=float, default=-2.0)
    ap.add_argument("--theta-max", type=float, default=5.0)
    ap.add_argument("--num-theta", type=int, default=1001)
    ap.add_argument("--Nint", type=int, default=4096, help="Number of ntegration points")
    ap.add_argument("--outfile", type=str, default="theta_vs_theta_mu_minus_logZ.png")
    args = ap.parse_args()

    if args.sigma <= 0:
        raise ValueError("sigma must be > 0")

    # Linear function
    def f(x, theta): 
        return theta * x 
    # Nonlinear function
    def f1(x, theta): 
        return np.sin(theta*x)


    theta = np.linspace(args.theta_min, args.theta_max, args.num_theta, args.Nint)
    U =  objective_theta(f, theta, args.mu, args.sigma)
    U1 = objective_theta(f1, theta, args.mu, args.sigma)
#    U-=np.max(U)
#    U1-=np.max(U1)

    # -------------------------------
    # Plot Results
    # -------------------------------
    plt.rc('text', usetex=True)
    plt.rc('font', size=22, family='serif', serif=['latin modern roman'])
    plt.rc('legend', fontsize=20)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
    
    plt.figure()
    plt.plot(theta, U, linewidth=2,label=r'$f(x)=\theta x$')
    plt.plot(theta, U1, linewidth=2,label=r'$f(x)=\sin(\theta x)$')
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\theta\mu - \log Z(\theta)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=200)
    plt.axis([np.min(theta), np.max(theta), 1.1 * np.min(U1), 1.1 * np.nanmax([U.max(), U1.max()])])
    
    # Sanity check calculation using Monte Carlo
    
    theta_test = -1.0
    Nmc = 2_000_000  # Monte Carlo samples

    # Draw Gaussian samples
    X_samples = np.random.normal(args.mu, args.sigma, size=Nmc)

    # First term: E[f(theta X)]
    Ef = np.mean(f1(X_samples, theta_test))

    # Second term: log E[exp(-f(theta X))]
    logEexp = np.log(np.mean(np.exp(f1(-X_samples, theta_test))))

    # Final result
    U = Ef - logEexp

    print(f"Theta = {theta_test}")
    print(f"Monte Carlo U(theta): {U:.8f}")
    
    
    plt.show()

