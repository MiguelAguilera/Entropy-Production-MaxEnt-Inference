#!/usr/bin/env python3
"""
Compute Z(theta) = E[ exp(-tanh(theta X)) ] for X ~ N(mu, sigma^2)
and plot theta vs (theta*mu - log Z(theta)).

Uses Gauss-Hermite quadrature:
    ∫_{-∞}^{∞} e^{-u^2} f(u) du ≈ Σ w_i f(u_i)
so that for X = mu + sqrt(2)*sigma*u,
    Z(theta) = (1/√π) ∫ e^{-u^2} exp(-tanh(theta*(mu + √2 σ u))) du.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss

from typing import Tuple

Array = np.ndarray

# ---------- ReLU family ----------
def relu(x: Array) -> Array:
    return np.maximum(x, 0.0)

def drelu(x: Array) -> Array:
    return (x > 0).astype(x.dtype)

def leaky_relu(x: Array, negative_slope: float = 0.01) -> Array:
    return np.where(x >= 0, x, negative_slope * x)

def dleaky_relu(x: Array, negative_slope: float = 0.01) -> Array:
    out = np.ones_like(x)
    out[x < 0] = negative_slope
    return out

def prelu(x: Array, a: float = 0.25) -> Array:
    # PReLU with scalar learnable 'a'
    return np.where(x >= 0, x, a * x)

def dprelu(x: Array, a: float = 0.25) -> Array:
    out = np.ones_like(x)
    out[x < 0] = a
    return out

# Randomized leaky ReLU (RReLU) — slope sampled per forward pass (usually per element or per channel)
def rrelu(x: Array, lower: float = 1/8, upper: float = 1/3, rng: np.random.Generator | None = None) -> Tuple[Array, Array]:
    """
    Returns (y, slope_used) so you can reuse the same slopes in backward.
    """
    rng = rng or np.random.default_rng()
    slope = rng.uniform(lower, upper, size=x.shape).astype(x.dtype)
    y = np.where(x >= 0, x, slope * x)
    return y, slope

def drrelu(x: Array, slope_used: Array) -> Array:
    out = np.ones_like(x)
    out[x < 0] = slope_used[x < 0]
    return out

# ---------- Sigmoid / Tanh ----------
def sigmoid(x: Array) -> Array:
    # Stable sigmoid
    pos = x >= 0
    neg = ~pos
    z = np.zeros_like(x)
    z[pos] = np.exp(-x[pos])
    z[neg] = np.exp(x[neg])
    out = np.empty_like(x)
    out[pos] = 1 / (1 + z[pos])
    out[neg] = z[neg] / (1 + z[neg])
    return out

def dsigmoid(x: Array) -> Array:
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x: Array) -> Array:
    return np.tanh(x)

def dtanh(x: Array) -> Array:
    t = np.tanh(x)
    return 1 - t * t

# ---------- Softplus ----------
def softplus(x: Array) -> Array:
    # Stable: log(1 + e^x) = max(x,0) + log1p(exp(-|x|))
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))

def dsoftplus(x: Array) -> Array:
    # derivative is sigmoid(x)
    return sigmoid(x)

# ---------- ELU ----------
def elu(x: Array, alpha: float = 1.0) -> Array:
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def delu(x: Array, alpha: float = 1.0) -> Array:
    out = np.ones_like(x)
    mask = x <= 0
    out[mask] = alpha * np.exp(x[mask])
    return out

# ---------- GELU ----------
# Exact (erf-based) version
def gelu(x: Array) -> Array:
    # x * Φ(x) with Φ via erf
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

# Tanh approximation (often used in practice)
def gelu_tanh(x: Array) -> Array:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def dgelu_approx(x: Array) -> Array:
    # derivative of tanh-approx GELU (good enough for custom backprop)
    c = np.sqrt(2/np.pi)
    x3 = x**3
    t = np.tanh(c * (x + 0.044715 * x3))
    dt = (1 - t**2) * c * (1 + 0.134145 * x**2)  # derivative of tanh inner
    return 0.5 * (1 + t) + 0.5 * x * dt

# helper for erf without importing scipy
def erf(x: Array) -> Array:
    # Use numpy.special if available; otherwise polynomial approx.
    # Prefer numpy.special.erf for accuracy:
    try:
        from numpy import special
        return special.erf(x)
    except Exception:
        # Fallback Abramowitz-Stegun approximation
        # (Good enough for small custom tests)
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        sign = np.sign(x)
        z = np.abs(x)
        t = 1.0 / (1.0 + p * z)
        y = 1 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-z*z)
        return sign * y

# ---------- Swish ----------
def swish(x: Array, beta: float = 1.0) -> Array:
    return x * sigmoid(beta * x)

def dswish(x: Array, beta: float = 1.0) -> Array:
    s = sigmoid(beta * x)
    return s + beta * x * s * (1 - s)

# ---------- Maxout ----------
def maxout(x: Array, groups: int) -> Tuple[Array, Array]:
    """
    x: shape (N, D), where D is divisible by groups (D = groups * k)
    returns (y, argmax_idx) where y is max across each group
    """
    N, D = x.shape
    assert D % groups == 0, "D must be divisible by groups"
    k = D // groups
    xg = x.reshape(N, groups, k)
    argmax_idx = np.argmax(xg, axis=2)
    y = np.take_along_axis(xg, argmax_idx[..., None], axis=2).squeeze(axis=2)
    return y, argmax_idx

def dmaxout(grad_y: Array, x: Array, groups: int, argmax_idx: Array) -> Array:
    """
    Backprop to x given grad_y. Returns grad_x with same shape as x.
    """
    N, D = x.shape
    k = D // groups
    grad = np.zeros_like(x.reshape(N, groups, k))
    # place grad_y into the winning index within each group
    rows = np.arange(N)[:, None]
    cols = np.arange(groups)[None, :]
    grad[rows, cols, argmax_idx] = grad_y
    return grad.reshape(N, D)

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

    # First term: E[f(theta X, theta)] over z
    Ef = np.sum(eff[:, None] * fvals, axis=0)     # (T,)

    # Second term: log E[exp(-f(theta X, theta))] via log-sum-exp along z
    g = -fvals                                    # (N,T)
    m = np.max(g, axis=0, keepdims=True)          # (1,T) pivot for stability
    logEexp = np.log(np.sum(eff[:, None] * np.exp(g - m), axis=0)) + m.squeeze(0)  # (T,)

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
        return theta * x  # bounded, smooth
    # Shifted tanh function
    def f1(x, theta): 
        return np.tanh(theta*x-1)  # bounded, smooth


    theta = np.linspace(args.theta_min, args.theta_max, args.num_theta, args.Nint)
    U =  objective_theta(f, theta, args.mu, args.sigma)
    U1 = objective_theta(f1, theta, args.mu, args.sigma)
#    U-=np.max(U)
#    U1-=np.max(U1)
    # Plot
    plt.figure()
    plt.plot(theta, U, linewidth=2,label=r'$f(x)=\theta x$')
    plt.plot(theta, U1, linewidth=2,label=r'$f(x)=\tanh(\theta x-1)$')
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

    # Compute f(X, theta) for all samples
    f_vals = f1(X_samples, theta_test)

    # First term: E[f(theta X)]
    Ef = np.mean(f_vals)

    # Second term: log E[exp(-f(theta X))]
    logEexp = np.log(np.mean(np.exp(-f_vals)))

    # Final result
    U = Ef - logEexp

    print(f"Theta = {theta_test}")
    print(f"Monte Carlo U(theta): {U:.8f}")
    
    
    plt.show()

