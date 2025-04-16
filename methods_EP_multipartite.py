import torch

import numpy as np
import sklearn 

# =======================
# Spin Model and Correlations
# =======================

def exp_EP_spin_model(Da, J, i):
    """
    Compute empirical entropy production contribution from spin `i` using 
    correlation matrix Da and interaction matrix J.
    """
    N, _ = J.shape
    return torch.sum((J[i, :] - J[:, i]) * Da) / 2

def correlations(S, T, i):
    """
    Compute pairwise correlations for spin `i`, averaged over Tetitions.
    """
    N, _ = S.shape
    Da = torch.einsum('r,jr->j', (-2 * S[i, :]), S) / T
<<<<<<< HEAD
#    Da[i] = 0  # zero out self-correlation
=======

    Da[i] = 0  # zero out self-correlation
>>>>>>> 47575bd086e08dd74d40c36d3b1ecc90ac463649
    return Da

def correlations4(S, T, i):
    """
    Compute 4th-order correlation matrix for spin `i`.
    """
    N, _ = S.shape
    K = (4 * S) @ S.T / T
#    K[i, :] = 0
#    K[:, i] = 0
    return K

# =======================
# Correlations with Theta (weighted by model parameters)
# =======================

def correlations_theta(S, T, theta, i):
    """
    Compute weighted pairwise correlations using theta.
    THESE ARE NOT YET DIVIDED BY THE NORMALIZATION CONSTANT.
    """
    N, _ = S.shape
    S_without_i = torch.cat((S[:i, :], S[i+1:, :]))  # remove spin i
    thf = (-2 * S[i, :]) * torch.matmul(theta, S_without_i)
    S1_S = -(-2 * S[i, :]) * torch.exp(-thf)
    Da = torch.einsum('r,jr->j', S1_S, S) / T
#    Da[i] = 0
    return Da

def correlations4_theta(S, T, theta, i):
    """
    Compute weighted 4th-order correlations using theta.
    THESE ARE NOT YET DIVIDED BY THE NORMALIZATION CONSTANT.
    """
    N, _ = S.shape
    S_without_i = torch.cat((S[:i, :], S[i+1:, :]))
    thf = (-2 * S[i, :]) * torch.matmul(theta, S_without_i)
    K = (4 * torch.exp(-thf) * S) @ S.T / T
#    K[i, :] = 0
#    K[:, i] = 0
    return K

# =======================
# Partition Function Estimate
# =======================

def norm_theta(S, T, theta, i):
    """
    Estimate normalization constant Z from the partition function under theta.
    """
    N, nflips = S.shape
    rep=T/N
    noF = rep - nflips
    S_without_i = torch.cat((S[:i, :], S[i+1:, :]))
    thf = (-2 * S[i, :]) * torch.matmul(theta, S_without_i)
    Z = (torch.sum(torch.exp(-thf)) / rep + noF / rep)
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

    I = torch.eye(Ks_no_diag_th.size(-1), dtype=Ks_th.dtype)
    
    alpha = 1e-1*torch.trace(Ks_no_diag_th)/len(Ks_no_diag_th)
    dtheta = torch.linalg.solve(Ks_no_diag_th + alpha*I, rhs_th)

    # epsilon = 1e-6
    # I = torch.eye(Ks_no_diag_th.size(-1), dtype=Ks_th.dtype)

    # while True:
    #     try:
    #         dtheta = torch.linalg.solve(Ks_no_diag_th + epsilon * I, rhs_th)
    #         break
    #     except torch._C._LinAlgError:
    #         epsilon *= 10  # Increase regularization if matrix is singular
    #         print(f"Matrix is singular, increasing epsilon to {epsilon}")

    return dtheta

# =======================
# Entropy Production Estimators
# =======================

def get_EP_Newton(S, T, i):
    """
    Compute entropy production estimate using the 1-step Newton method and the MTUR method for spin i.
    """
    N, _ = S.shape
    Da = correlations(S, T, i)
    Ks = correlations4(S, T, i)
    Ks -= torch.einsum('j,k->jk', Da, Da)


    
    theta = solve_linear_theta(Da, -Da, Ks, i)
    Dai = remove_i(Da, i)
    
    Z = norm_theta(S, T, theta, i)
    if i ==N-1:
        print(Da*N)
        print(Z)

    sig_MTUR = (theta * Dai).sum()

    Dai = remove_i(Da, i)
<<<<<<< HEAD
    sig_N1 = (theta * Dai).sum() - torch.sum(torch.log(norm_theta(S, T, theta, i)))/N
=======
    sig_N1 = (theta * Dai).sum() - torch.log(norm_theta(S, T, theta, i))
>>>>>>> 47575bd086e08dd74d40c36d3b1ecc90ac463649
    return sig_N1, sig_MTUR, theta, Da

def get_EP_Newton2(S, T, theta_lin, Da, i):
    """
    One iteration of Newton-Raphson to refine theta estimation.
    """
    N, _ = S.shape
    Da_th = correlations_theta(S, T, theta_lin, i)
    Ks_th = correlations4_theta(S, T, theta_lin, i)


    Z = norm_theta(S, T, theta_lin, i)
    if i ==N-1:
        print(Da_th*N)
        print(Ks_th[-1,:]*N)
        print(Z)
    Da_th /= Z
    Ks_th = Ks_th / Z - torch.einsum('j,k->jk', Da_th, Da_th)

    theta_lin2 = solve_linear_theta(Da, Da_th, Ks_th, i)
    theta = theta_lin + theta_lin2

    Dai = remove_i(Da, i)
    sig_N2 = (theta * Dai).sum() - torch.log(norm_theta(S, T, theta, i))/N
    return sig_N2, theta_lin2

