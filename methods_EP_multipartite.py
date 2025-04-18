import torch

import numpy as np

# =======================
# Spin Model and Correlations
# =======================

def exp_EP_spin_model(Da, J_i, i):
    """
    Compute empirical entropy production contribution from spin `i` using 
    correlation matrix Da and interaction matrix J.
    """
#    return torch.sum((J[i, :]-J[:,i])* Da)/2
    return torch.sum(J_i * Da) 

def correlations(S, i):
    """
    Compute pairwise correlations for spin `i`, averaged over Tetitions.
    """
    N, nflips = S.shape
    Da = torch.einsum('r,jr->j', (-2 * S[i, :]), S) / nflips
    return Da

def correlations4(S, i):
    """
    Compute 4th-order correlation matrix for spin `i`.
    """
    N, nflips = S.shape
    K = (4 * S) @ S.T / nflips
    return K

# =======================
# Correlations with Theta (weighted by model parameters)
# =======================

def correlations_theta(S, theta, i):
    """
    Compute weighted pairwise correlations using theta.
    THESE ARE NOT YET DIVIDED BY THE NORMALIZATION CONSTANT.
    """
    N, nflips = S.shape
    S_without_i = torch.cat((S[:i, :], S[i+1:, :]))  # remove spin i
    thf = (-2 * S[i, :]) * torch.matmul(theta, S_without_i)
    S1_S = -(-2 * S[i, :]) * torch.exp(-thf)
    Da = torch.einsum('r,jr->j', S1_S, S) / nflips
    Z = torch.sum(torch.exp(-thf)) / nflips
    return Da, Z

def correlations4_theta(S, theta, i):
    """
    Compute weighted 4th-order correlations using theta.
    THESE ARE NOT YET DIVIDED BY THE NORMALIZATION CONSTANT.
    """
    N, nflips = S.shape
    S_without_i = torch.cat((S[:i, :], S[i+1:, :]))
    thf = (-2 * S[i, :]) * torch.matmul(theta, S_without_i)
    K = (4 * torch.exp(-thf) * S) @ S.T / nflips
#    K[i, :] = 0
#    K[:, i] = 0
    return K

# =======================
# Partition Function Estimate
# =======================

def norm_theta(S, theta, i):
    """
    Estimate normalization constant Z from the partition function under theta.
    """
    N, nflips = S.shape
    S_without_i = torch.cat((S[:i, :], S[i+1:, :]))
    thf = (-2 * S[i, :]) * torch.matmul(theta, S_without_i)
    Z = torch.sum(torch.exp(-thf)) / nflips
    return Z

# =======================
# Matrix Processing Utilities
# =======================

def K_nodiag(Ks, i):
    """
    Remove the i-th row and column from matrix Ks.
    """
    Ks_no_row = torch.cat([Ks[:i, :], Ks[i+1:, :]], dim=0)
    Ks_no_row_col = torch.cat([Ks_no_row[:, :i], Ks_no_row[:, i+1:]], dim=1)
    return Ks_no_row_col

def remove_i(A, i):
    """
    Remove the i-th element from a 1D tensor A.
    """
    r = torch.cat((A[:i], A[i+1:]))
    #print(A,i,r)
    return r

# =======================
# Linear Solver for Theta Estimation
# =======================

def solve_linear_theta(Da, Da_th, Ks_th, i):
    """
    Solve the linear system to compute theta using regularized inversion.
    """
    Dai    = remove_i(Da, i)
    Dai_th = remove_i(Da_th, i)
    Ks_no_diag_th = K_nodiag(Ks_th, i)

    rhs_th = Dai - Dai_th

#    I = torch.eye(Ks_no_diag_th.size(-1), dtype=Ks_th.dtype)
#    
#    alpha = 1e-1*torch.trace(Ks_no_diag_th)/len(Ks_no_diag_th)
#    dtheta = torch.linalg.solve(Ks_no_diag_th + alpha*I, rhs_th)

    epsilon = 1e-6
    I = torch.eye(Ks_no_diag_th.size(-1), dtype=Ks_th.dtype)

    while True:
        try:
            dtheta = torch.linalg.solve(Ks_no_diag_th + epsilon * I, rhs_th)
            break
        except torch._C._LinAlgError:
            epsilon *= 10  # Increase regularization if matrix is singular
            print(f"Matrix is singular, increasing epsilon to {epsilon}")

    return dtheta

# =======================
# Entropy Production Estimators
# =======================
# import gd
# def get_EP_gd2(S, i, x0=None):
#     # NEED TO FINISH THIS
#     N = S.shape[0]
#     if x0 is None:
#         x0 = np.zeros(N)

#     g_t = torch.zeros_like(S)
#     for j in range(N):
#         g_t[j,:] = -2*S[i,:]*S[j,:]    # these are the observables
#     g_t_noI = torch.cat((g_t[:i,:], g_t[i+1:,:]))
#     g_avg=g_t_noI.mean(axis=1)           # conditional averages 

#     def func(theta): 
#         obj = theta@g_avg - torch.log(norm_theta(S, theta, i))
#         return -obj
#     def grad(theta):
#         m1 = g_avg
#         Da_th = correlations_theta(S, theta, i)/norm_theta(S, theta, i)
#         r = m1
#         r[:i] -= Da_th[:i]
#         r[i:] -= Da_th[i+1:]
#         return r

#     optimal_params, obj_values, iterations = gd.adam_optimizer(
#          func, 
#          grad,
#          x0
#     )
#     return obj_values[-1], optimal_params

def get_EP_Newton(S, i):
    """
    Compute entropy production estimate using the 1-step Newton method for spin i.
    """
    N, nflips = S.shape
    Da = correlations(S, i)
    Ks = correlations4(S, i)
    Ks -= torch.einsum('j,k->jk', Da, Da)
    
    theta = solve_linear_theta(Da, -Da, Ks, i)
    Dai = remove_i(Da, i)
    
    Z = norm_theta(S, theta, i)

    Dai = remove_i(Da, i)
    sig_N1 = theta @ Dai - torch.log(norm_theta(S, theta, i))
    return sig_N1, theta, Da



def get_EP_MTUR(S, i):
    """
    Compute entropy production estimate using the MTUR method for spin i.
    """
    Da = correlations(S, i)
    Ks = correlations4(S, i)
    theta = solve_linear_theta(Da, -Da, Ks, i)
    Dai = remove_i(Da, i)
    sig_MTUR = theta @ Dai
    return sig_MTUR



def get_EP_Newton2(S, theta_init, Da, i, delta=0.5):
    """
    Perform one iteration of a constrained Newton-Raphson update to refine the parameter theta.

    Parameters:
    -----------
    S : torch.Tensor
        Binary spin configurations of shape (N, nflips).
    theta_init : torch.Tensor
        Current estimate of the parameter vector theta (with zero at index i).
    Da : torch.Tensor
        Empirical first-order correlations (e.g., ⟨s_j⟩).
    i : int
        Index of the spin being updated (excluded from optimization).
    delta : float or None, optional
        If provided, sets a maximum relative norm for the update step 
        (default: 1.0, meaning ||Δθ|| ≤ delta * ||θ||).

    Returns:
    --------
    sig_N2 : float
        Updated estimate of the log-partition function contribution (e.g., entropy production).
    delta_theta : torch.Tensor
        Computed Newton step (Δθ).
    """

    N, nflips = S.shape

    # Compute model-averaged first-order and fourth-order statistics (excluding index i)
    Da_th, Z = correlations_theta(S, theta_init, i)
    Ks_th = correlations4_theta(S, theta_init, i)

    # Normalize by partition function Z
    Da_th /= Z
    Ks_th = Ks_th / Z - torch.einsum('j,k->jk', Da_th, Da_th)  # Covariance estimate

    # Compute Newton step: Δθ = H⁻¹ (Da - Da_th)
    delta_theta = solve_linear_theta(Da, Da_th, Ks_th, i)

    # Optional: constrain the step to be within a trust region
    if delta is not None:
        max_step = delta * torch.norm(theta_init)
        step_norm = torch.norm(delta_theta)
        if step_norm > max_step:
            delta_theta = delta_theta * (max_step / step_norm)

    # Apply the update
    theta = theta_init + delta_theta

    # Remove index i from Da for calculating log-partition contribution
    Dai = remove_i(Da, i)

    # Compute surrogate objective (e.g., log-partition or entropy production)
    sig_N2 = (theta * Dai).sum() - torch.log(norm_theta(S, theta, i))

    return sig_N2, theta
    
def get_EP_Adam(S, theta_init, Da, i, num_iters=100, 
                     beta1=0.9, beta2=0.999, lr=0.1, eps=1e-8, 
                     tol=1e-4, skip_warm_up=False):
    """
    Performs multiple Adam-style updates to refine theta estimation.
    
    Arguments:
        S         : binary spin samples, shape (N, num_flips)
        theta_init: initial theta vector
        Da        : empirical expectation vector
        i         : index to remove from theta (current spin)
        num_iters : number of Adam updates
        beta1, beta2: Adam moment decay parameters
        lr        : learning rate
        eps       : epsilon for numerical stability
        tol       : tolerance for early stopping
    
    Returns:
        sig_N2    : final entropy production estimate
        delta_all : total change in theta (final - initial)
        theta     : final updated theta
    """
    theta = theta_init.clone()
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    

    for t in range(1, num_iters + 1):
        Da_th, Z = correlations_theta(S, theta, i)
        Da_th /= Z

        grad = remove_i(Da - Da_th, i)

        # Adam moment updates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad.pow(2)
        if skip_warm_up:
            m_hat = m
            v_hat = v
        else:
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

        # Compute parameter update
        delta_theta = lr * m_hat / (v_hat.sqrt() + eps)

        # Apply update to full theta
        theta += delta_theta

        # Optional: early stopping
        if delta_theta.norm() < tol:
            break

    Dai = remove_i(Da, i)
    sig_Adam = (theta * Dai).sum() - torch.log(norm_theta(S, theta, i))
    return sig_Adam, theta

