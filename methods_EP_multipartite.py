import torch
import numpy as np
import time

# =======================
# Spin Model and Correlations
# =======================

def exp_EP_spin_model(Da, J_i, i):
    """
    Compute empirical entropy production contribution from spin `i` using 
    correlation matrix Da and interaction matrix J.
    """
    assert J_i.dim() == 1, f"Tensor must be 1D, but got {J_i.dim()}D"
    return (J_i @ Da).item()
    
def correlations(S, i):
    """
    Compute pairwise correlations for spin `i`
    """
    nflips, N = S.shape
    Da = (-2 * S[:, i]) @ S / nflips 
    return Da

def correlations4(S, i, num_chunks=20):
    """
    Compute 4th-order correlation matrix for spin `i`.
    
    If `num_chunks` is provided, computes the matrix in `num_chunks` pieces to save memory.
    """
    nflips, N = S.shape

    if num_chunks is None:
        # Normal full matrix multiplication
        K = (4 * S.T) @ S / nflips
    else:
        # Chunked multiplication
        device = S.device
        K = torch.zeros((N, N), device=device)

        chunk_size = (nflips + num_chunks - 1) // num_chunks  # Ceiling division

        for start in range(0, nflips, chunk_size):
            end = min(start + chunk_size, nflips)
            S_chunk = S[start:end, :]  # (chunk_size, N)

            K += (4 * S_chunk.T) @ S_chunk

            del S_chunk
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        K /= nflips

    return K

# =======================
# Correlations with Theta (weighted by model parameters)
# =======================

def correlations_theta(S, theta, i):
    """
    Compute weighted pairwise correlations using theta.
    """
    nflips, N = S.shape
#    S_without_i = torch.cat((S[:, :i], S[:, i+1:]), dim=1)  # remove spin i
    theta_padded = add_i(theta,i)
    th_g = (-2 * S[:, i]) * (S @ theta_padded)
    th_g_min = torch.min(th_g)    # substract max of -th_g
    S1_S = -(-2 * S[:, i]) * torch.exp(-th_g+th_g_min)
    Da = S1_S @ S / nflips
    Z = torch.sum(torch.exp(-th_g+th_g_min)) / nflips
    return Da/Z, Z*torch.exp(-th_g_min)

def correlations4_theta(S, theta, i, num_chunks=20):
    """
    Compute weighted 4th-order correlations using theta.
    
    If `num_chunks` is provided, computes the matrix in chunks to save memory.
    """
    nflips, N = S.shape
    device = S.device

    # Pad theta with a zero at position i
    theta_padded = torch.cat((theta[:i], torch.tensor([0.0], device=device), theta[i:]))


    th_g = (-2 * S[:, i]) * (S @ theta_padded)
    th_g_min = torch.min(th_g)    # substract max of -th_g
    Z = torch.sum(torch.exp(-th_g+th_g_min)) / nflips
    
    if num_chunks is None:
        # Full computation
        th_g = (-2 * S[:, i]) * (S @ theta_padded)
        K = (4 * torch.exp(-th_g+th_g_min) * S.T) @ S / nflips
    else:
        # Chunked computation
        K = torch.zeros((N, N), device=device)
        chunk_size = (nflips + num_chunks - 1) // num_chunks  # Ceiling division

        for r in range(num_chunks):
            start = r * chunk_size
            end = min((r + 1) * chunk_size, nflips)
            S_chunk = S[start:end]
            
            th_g_chunk = (-2 * S_chunk[:, i]) * (S_chunk @ theta_padded)
            K += (4 * torch.exp(-th_g_chunk+th_g_min) * S_chunk.T) @ S_chunk

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        K /= nflips

    return K/Z

# =======================
# Partition Function Estimate
# =======================

def norm_theta(S, theta, i):
    """
    Estimate normalization constant Z from the partition function under theta.
    """
    nflips, N = S.shape
#    S_without_i = torch.cat((S[:, :i], S[:, i+1:]), dim=1)  # remove spin i
    theta_padded = add_i(theta,i)
    th_g = (-2 * S[:, i]) * (S @ theta_padded)
    Z = torch.sum(torch.exp(-th_g)) / nflips
    return Z


def log_norm_theta(S, theta, i):
    """
    Estimate normalization constant Z from the partition function under theta.
    """
    nflips, N = S.shape
#    S_without_i = torch.cat((S[:, :i], S[:, i+1:]), dim=1)  # remove spin i
    theta_padded = add_i(theta,i)
    th_g = (-2 * S[:, i]) * (S @ theta_padded)
    th_g_min = torch.min(th_g)    # substract max of -th_g
    logZ = torch.log(torch.sum(torch.exp(-th_g+th_g_min)) / nflips) - th_g_min
    return logZ
    

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
    return r

# =======================
# Linear Solver for Theta Estimation
# =======================

def solve_linear_theta(Da, Da_th, Ks_th, i, eps=1e-5, method='QR'):
    """
    Solve the linear system to compute theta using regularized inversion.
    """

    assert not torch.isnan(Da).any() 
    assert not torch.isnan(Da_th).any() 
    assert not torch.isnan(Ks_th).any() 

    Dai    = remove_i(Da, i)
    Dai_th = remove_i(Da_th, i)
    Ks_no_diag_th = K_nodiag(Ks_th, i)
    rhs_th = Dai - Dai_th

#    I = torch.eye(Ks_no_diag_th.size(-1), dtype=Ks_th.dtype)
#    
#    alpha = 1e-1*torch.trace(Ks_no_diag_th)/len(Ks_no_diag_th)
#    dtheta = torch.linalg.solve(Ks_no_diag_th + alpha*I, rhs_th)

#    return torch.linalg.solve(Ks_no_diag_th, rhs_th)

    I = torch.eye(Ks_no_diag_th.size(-1), dtype=Ks_th.dtype, device=Ks_th.device)
    if method=='LS':
        dtheta = torch.linalg.lstsq(Ks_no_diag_th + eps * I, rhs_th).solution
    else:
        dtheta = None
        while True:
            if eps > 1:
                print('Something is wrong, cannot regularize enough')
                return torch.nan + torch.zeros_like(Dai)
            try:
                if method=='solve':
                    dtheta = torch.linalg.solve(Ks_no_diag_th + eps * I, rhs_th)
                elif method=='QR':
                    Q, R = torch.linalg.qr(Ks_no_diag_th + eps * I)
                    # dtheta = torch.linalg.solve(R, Q.T @ rhs_th)
                    dtheta = torch.linalg.solve_triangular( R, (Q.T @ rhs_th).unsqueeze(1), upper=True).squeeze()

                if dtheta is not None and not torch.isinf(dtheta).any() and not torch.isnan(dtheta).any():
                    break
            except torch._C._LinAlgError:
                print(f"Matrix is singular, increasing epsilon {eps} by 10")
            eps *= 10  
        
    return dtheta


def get_EP_Newton(S, i, num_chunks=None):
    """
    Compute entropy production estimate using the 1-step Newton method for spin i.
    """
    Da = correlations(S, i)
    Ks = correlations4(S, i,num_chunks)
    Ks -= torch.einsum('j,k->jk', Da, Da)
    theta = solve_linear_theta(Da, -Da, Ks, i)
    Dai = remove_i(Da, i)
    
    Z=0
    eps = 1e-4
    for _ in range(100):  # regularize until  we get a non-zero Z
        theta = solve_linear_theta(Da, -Da, Ks, i, eps=eps)
        Z = norm_theta(S, theta, i)
        if not np.isclose(Z.item(),0) and not np.isinf(Z.item()):
            break
        eps *= 10
    else:
        print('get_EP_Newton cannot regularize enough!')
        return np.nan, theta, Da

    sig_N1 = theta @ Dai - torch.log(Z)
    v = sig_N1.item()
    if np.isinf(v):
        print('get_EP_newton, v is inf', theta, Da, Z, theta @ Dai)

    return v, theta, Da


def get_EP_MTUR(S, i,num_chunks=None):
    """
    Compute entropy production estimate using the MTUR method for spin i.
    """
    Da = correlations(S, i)
    Ks = correlations4(S, i, num_chunks)
    theta = solve_linear_theta(Da, -Da, Ks, i)
    Dai = remove_i(Da, i)
    sig_MTUR = theta @ Dai
    return sig_MTUR.item()



def get_EP_Newton2(S, theta_init, Da, i, delta=None, th=0.5, num_chunks=None):
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

    # Compute model-averaged first-order and fourth-order statistics (excluding index i)
    Da_th = correlations_theta(S, theta_init, i)
    Ks_th = correlations4_theta(S, theta_init, i,num_chunks)

    # Transform into covariance
    Ks_th = Ks_th - torch.einsum('j,k->jk', Da_th, Da_th)  # Covariance estimate
    
    if torch.isinf(Ks_th).any():
        # Error occured, usually means theta is too big
        return np.nan, theta_init*np.nan

    # Compute Newton step: Δθ = H⁻¹ (Da - Da_th)
    delta_theta = solve_linear_theta(Da, Da_th, Ks_th, i)

    # Optional: constrain the step to be within a trust region
    if delta is not None:
        max_step = delta * torch.norm(theta_init)
        step_norm = torch.norm(delta_theta)
        if step_norm > max_step:
            delta_theta = delta_theta * (max_step / step_norm)
        theta = theta_init + delta*delta_theta
    else:
        Dai = remove_i(Da, i)
        Dai_th = remove_i(Da_th, i)
        alpha = 1
        d1 = (delta_theta @ Dai_th).item()
        d2 = (delta_theta @ (Dai-Dai_th)).item() / 2
        dlogZ = alpha * d1 + alpha**2 * d2
#        print(dlogZ)
        while np.abs(dlogZ)>th:
            alpha *= 0.95
            dlogZ = alpha * d1 + alpha**2 * d2
        theta = theta_init + alpha*delta_theta
    # Remove index i from Da for calculating log-partition contribution


    
#    if delta is not None:
#        max_step = delta * torch.norm(theta_init)
#        step_norm = torch.norm(delta_theta)
#        if step_norm > max_step:
#            delta_theta = delta_theta * (max_step / step_norm)
##    print('delta',delta, delta_theta @ (2*Dai_th-Dai))
#    theta = theta_init + delta*delta_theta

    # Compute surrogate objective (e.g., log-partition or entropy production)
    Dai = remove_i(Da, i)
    sig_N2 = (theta * Dai).sum() - log_norm_theta(S, theta, i)

    return sig_N2.item(), theta




    
def get_EP_Newton_steps(S, theta_init, sig_init, Da, i, num_chunks=None, tol=1e-3, max_iter=10):
    nflips,N = S.shape
    sig_old = sig_init
    theta_N = theta_init.clone()
    dsig = sig_old
    count = 0
    eps = 1e-8  # small epsilon to avoid division by zero
    sig_new = sig_init
    while count < max_iter:
    
        sig_old = sig_new
        theta_old = theta_N.clone()
        sig_new = np.nan
        sig_new, theta_N = get_EP_Newton2(S, theta_N.clone(), Da, i, num_chunks=num_chunks,delta=None)

        dsig = sig_new - sig_old
        count += 1
        rel_change = np.abs(dsig) / (np.abs(sig_old) + eps)
#        Z= log_norm_theta(S, theta_N, i)
        if rel_change <= tol:# or Z>0.1:
            break

        #print(sig_new, np.log(nflips))
        if sig_new > np.log(nflips):
            #print(f'Break at iteration {count}: log(nflips)={np.log(nflips):.4e}, sig_new={sig_new:.4e}')
            break 
        if sig_new < sig_old or np.isnan(sig_new):
            # print(f'Break at iteration {count}: sig_old={sig_old:.4e}, sig_new={sig_new:.4e}')
            return sig_new, theta_N
        
    return sig_new, theta_N


def get_EP_Newton_steps_holdout(S, i, num_chunks=None, tol=1e-3, max_iter=50):

    def newton_step_trn(theta_init, delta=.25):
        Da_th, Z = correlations_theta(S_trn, theta_init, i)
        Ks_th    = correlations4_theta(S_trn, theta_init, i) / Z

        Da_th   /= Z
        Ks_th    = Ks_th - torch.einsum('j,k->jk', Da_th, Da_th)  # Covariance estimate
    
        if torch.isinf(Ks_th).any():
            return np.nan, theta_init*np.nan

        delta_theta = solve_linear_theta(Da, Da_th, Ks_th, i)

        if delta is not None:
             max_step = delta * torch.norm(theta_init)
             step_norm = torch.norm(delta_theta)
             if step_norm > max_step:
                 delta_theta = delta_theta * (max_step / step_norm)


        theta = theta_init + delta_theta

        sig   = theta @ Dai_trn - torch.log(norm_theta(S, theta, i))

        return sig.item(), theta_init + delta_theta


    nflips = int(S.shape[0]/2)
    # nflips = S.shape[0]
    S_trn = S[:nflips,:]
    S_tst = S[nflips:,:]
    Dai_trn = remove_i(correlations(S_trn,i),i)
    Dai_tst = remove_i(correlations(S_tst,i),i)

    def get_test_objective(theta):
        return (theta @ Dai_tst - log_norm_theta(S_tst, theta, i)).item()

    sig_new_trn, theta_N, Da = get_EP_Newton(S_trn, i)
    #sig_new, theta_N  = get_EP_Newton2(S, theta_init, Da, i, num_chunks=num_chunks)
    sig_new_tst = get_test_objective(theta_N)
    sig_old_trn = sig_old_tst = np.nan 
    dsig_trn    = dsig_tst    = np.nan

    count = 0
    sig_N = sig_new_tst
    eps = 1e-8  # small epsilon to avoid division by zero

    if np.isnan(sig_new_tst) or np.isinf(sig_new_tst):
        return sig_new_trn, theta_N 

    while count < max_iter:
        dsig_trn = sig_new_trn - sig_old_trn
        dsig_tst = sig_new_tst - sig_old_tst

        rel_change_trn = np.abs(dsig_trn) / (np.abs(sig_old_trn) + eps)
        rel_change_tst = np.abs(dsig_tst) / (np.abs(sig_old_tst) + eps)

        if rel_change_trn <= tol or rel_change_tst <= tol:
            sig_N = sig_old_tst
            break

        if sig_new_tst > np.log(nflips) or sig_new_trn > np.log(nflips):
            #print(f'Break at iteration {count}: log(nflips)={np.log(nflips):.4e}, sig_new={sig_new:.4e}')
            sig_N = sig_old_tst
            break 

        if sig_new_tst < sig_old_tst or sig_new_trn < sig_old_trn: # early stopping
            sig_N = sig_old_tst
            break

        if np.isnan(sig_new_tst) or np.isnan(sig_new_trn) or np.isinf(sig_new_tst) or np.isinf(sig_new_trn):
            #print(f'Break at iteration {count}: sig_old={sig_old:.4e}, sig_new={sig_new:.4e}')
            sig_N = sig_old_tst
            break


        sig_old_trn = sig_new_trn
        sig_new_trn, theta_N = newton_step_trn(theta_N)
        
        sig_old_tst = sig_new_tst
        sig_new_tst = get_test_objective(theta_N)


        sig_N    = sig_new_tst
        count   += 1

    # Dai = remove_i(correlations(S,i),i)
    # sig_N = (theta_N @ Dai - torch.log(norm_theta(S, theta_N, i))).item()
    return sig_N, theta_N    

            
def get_EP_BFGS(S, theta_init, Da, i, alpha=1., delta=0.05, max_iter=10, tol=1e-6):
    """
    Manual BFGS maximization for f(θ) = θ^T ⟨g⟩ - log Z(θ)

    Parameters
    ----------
    S : torch.Tensor
        Binary spin configurations (N, nflips).
    theta_init : torch.Tensor
        Initial estimate of θ (with θ[i] = 0 fixed).
    Da : torch.Tensor
        Empirical averages ⟨g⟩ (e.g., ⟨s_j⟩).
    i : int
        Index of fixed spin variable.
    delta : float
        Trust region parameter.
    max_iter : int
        Maximum number of BFGS iterations.
    tol : float
        Gradient norm tolerance for convergence.

    Returns
    -------
    sig_BFGS : float
        Final surrogate objective value.
    theta : torch.Tensor
        Final estimate of θ.
    """

        
    theta = theta_init.clone()
    n = theta.numel()
    
    Dai = remove_i(Da, i)
    I = torch.eye(n)
    
    # Initial model estimate and gradient
    Da_th, = correlations_theta(S, theta, i)
    Da_th_wo_i = remove_i(Da_th, i)
    grad = Dai - Da_th_wo_i

    H = torch.eye(n)  # initial inverse Hessian approximation

    for it in range(max_iter):
        if grad.norm() < tol:
            break

        # Compute direction and step
        p = H @ grad
        if grad.norm() > delta * theta.norm():
            p = p * (delta * theta.norm() / grad.norm())
        theta_new = theta + alpha * p

        # Compute new gradient
        Da_th_new= correlations_theta(S, theta_new, i)
        grad_new = Dai - remove_i(Da_th_new, i)

        # BFGS update
        s = theta_new - theta
        y = grad_new - grad
        rho = 1.0 / (y @ s + 1e-10)

        H = (I - rho * s[:, None] @ y[None, :]) @ H @ (I - rho * y[:, None] @ s[None, :]) + rho * s[:, None] @ s[None, :]

        # Reuse values for next iteration
        theta = theta_new
        grad = grad_new
        Da_th = Da_th_new  # optional, for final objective


    # Final update
    p = H @ grad_new  # ascent direction
    if grad.norm() > delta * theta.norm():
        p = p * (delta * theta.norm() / grad.norm())
    theta = theta + alpha * p

    # Final objective
    sig_BFGS = (theta * Dai).sum() - tlog_norm_theta(S, theta, i)

    return sig_BFGS.item(), theta
    
def get_objective_numpy(S_i, Da, theta,i):
    # Perform calculation using float64 arithmetic for more accuracy
    Dai      = remove_i(Da, i).cpu().numpy().astype('float64')
    theta_np = theta.cpu().numpy().astype('float64')
    Snp      = S_i.cpu().numpy().astype('float64')
    S_only_i = Snp[:,i]
    S_without_i = np.hstack((Snp[:, :i], Snp[:, i+1:]))
    X = -2 * S_only_i[:,None]*S_without_i
    th_g = X@theta_np
    Z = np.exp(-th_g).mean()
    return theta_np @ Dai - np.log(Z)


def add_i(x,i):
    return torch.cat((x[:i], torch.tensor([0.0], device=x.device), x[i:]))   

def get_EP_Adam(S_i, Da, theta_init, i, num_iters=1000, 
                     beta1=0.9, beta2=0.999, lr=0.01, eps=1e-8, 
                     tol=1e-4, skip_warm_up=False,
                     timeout=60):
    """
    Performs multiple Adam-style updates to refine theta estimation.
    
    Arguments:
        S         : binary spin samples, shape (num_flips,N)
        theta_init: initial theta vector
        Da        : empirical expectation vector
        i         : index to remove from theta (current spin)
        num_iters : number of Adam updates
        beta1, beta2: Adam moment decay parameters
        lr        : learning rate
        eps       : epsilon for numerical stability
        tol       : tolerance for early stopping
        skip_warm_up : Adam option
        timeout : maximum second to run
    
    Returns:
        sig_gd    : final entropy production estimate
        theta     : final updated theta
    """

    S_i    = S_i.T.contiguous()   # Transpose for quicker calculations

    nflips = S_i.shape[1]

    theta = add_i(theta_init, i)

    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    N = len(theta)

    stime = time.time()
    twice_S_onlyi = 2*S_i[i,:]

    X = (-torch.einsum('j,ij->ij', twice_S_onlyi, S_i)).contiguous() # S_onlyi[:,None] * S_i
    # -2 * S_onlyi * S_i # 
    last_val = -np.inf
    cur_val  = torch.tensor(np.nan)

    for t in range(1, num_iters + 1):
        th_g = theta@X
        
        Y = torch.exp(-th_g)
        Z = torch.mean(Y)
        S1_S = twice_S_onlyi * Y # -(-2 * S_onlyi) * Y

        #print(S1_S.shape, S_i.shape)

        Da_th = torch.einsum('r,jr->j', S1_S, S_i) / nflips / Z #  @torch.einsum('r,rj->j', S1_S, S_i) / nflips
        #Da_th /= Z

        cur_val = (theta @ Da - torch.log(Z)).item()
        #return 0, theta_init

        if np.isnan(cur_val):
            break
        if cur_val > np.log(nflips):
            break
        # Early stopping
        if t>5 and ((time.time()-stime > timeout) or (np.abs((last_val - cur_val)/(last_val+1e-8)) < tol)):
            break

        last_val = cur_val

        grad = Da - Da_th

        if False:
            # regular gradient descent
            theta += lr * grad
            
        else:
            # Adam moment updates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            if skip_warm_up:
                m_hat = m
                v_hat = v
            else:
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

            # Compute parameter update
            delta_theta = lr * m_hat / (v_hat.sqrt() + eps)

            theta += delta_theta

        theta[i]=0.0

    return cur_val, torch.cat((theta[:i], theta[i+1:]))



