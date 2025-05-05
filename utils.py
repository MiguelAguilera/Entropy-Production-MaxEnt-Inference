# Includes various handy functions
import os
import numpy as np
import warnings

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable fallback for MPS backend
import torch

def batch_outer(X, num_chunks=None):
    # Compute outer product X @ X.T/nrow. Do it in batches if requested for lower memory requirements
    nrow, ncol = X.shape
    if num_chunks is None:
        K = X@X.T
    else:
        # Chunked computation, sometimes helpful for memory reasons
        K = torch.zeros((ncol, ncol), device=X.device)
        chunk_size = (nrow + num_chunks - 1) // num_chunks  # Ceiling division

        for r in range(num_chunks):
            start = r * chunk_size
            end = min((r + 1) * chunk_size, ncol)
            g_chunk = X[start:end]
            K += g_chunk @ g_chunk.T
    return K/nrow


# Helpful tensor processing functions

def eye_like(A):    # Returns torch identity matrix with same dimensions, data type, and device as A
    assert(A.ndim == 2 and A.shape[0] == A.shape[1])
    return torch.eye(A.shape[0], dtype=A.dtype, device=A.device)

def is_infnan(x): # return True if x is either infinite or NaN
    x = float(x)
    return np.isinf(x) or np.isnan(x)

def remove_i(x, i):  # Helpful function to remove the i-th element from a 1d tensor x.
    r = torch.cat((x[:i], x[i+1:]))
    return r

def numpy_to_torch(X):  # Convert numpy array to torch tensor if needed
    if not isinstance(X, torch.Tensor): 
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X.astype('float32')).to(torch.get_default_device()).contiguous()
        else:
            raise Exception("Argument must be a torch tensor or numpy array")
    return X


# Torch stuff

def set_default_torch_device():
    # Determines the best available device for PyTorch operations and sets it as default.
    # Returns the torch device that was set as default ('mps', 'cuda', or 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.set_default_device(device)
    warnings.filterwarnings("ignore", message="The operator 'aten::_linalg_solve_ex.result' is not currently supported on the MPS backend and will fall back to run on the CPU", category=UserWarning)
    return device


def empty_torch_cache():  # Empty torch cache
    if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
        torch.cuda.empty_cache()
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()

def torch_synchronize():  # Empty torch cache
    if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
        torch.cuda.synchronize()
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.synchronize()

# ****** Linear algebra stuff **********

def steihaug_toint_cg(A, b, trust_radius, tol=1e-10, max_iter=None):
    """
    Steihaug-Toint Conjugate Gradient method for approximately solving
    min_x 0.5 x^T A x - b^T x  subject to ||x|| <= trust_radius
    where A is symmetric (not necessarily positive definite).
    
    Args:
        A (torch.Tensor): Symmetric matrix (n x n).
        b (torch.Tensor): Right-hand side vector (n).
        trust_radius (float): Trust region radius.
        tol (float): Tolerance for convergence on residual norm.
        max_iter (int): Maximum number of iterations.
    
    Returns:
        torch.Tensor: Approximate solution x.
    """
    def find_tau(x, d, trust_radius):
        # Solve quadratic equation for tau: ||x + tau*d||^2 = trust_radius^2
        a = d @ d
        b_lin = 2 * (x @ d)
        c = (x @ x) - trust_radius**2
        discriminant = b_lin**2 - 4 * a * c
        if discriminant < 0:
            discriminant = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        sqrt_discriminant = torch.sqrt(discriminant)
        tau = (-b_lin + sqrt_discriminant) / (2 * a)
        return tau

    n = b.shape[0]
    max_iter = max_iter or 10 * n
    x = torch.zeros_like(b)
    r = b.clone()
    d = r.clone()

    if r.norm() < tol:
        return x

    for k in range(max_iter):
        Hd = A @ d
        dHd = d @ Hd

        if dHd <= 0: # Negative curvature detected → move to boundary
            tau = find_tau(x, d, trust_radius)
            return x + tau * d

        alpha = (r @ r) / dHd
        x_next = x + alpha * d

        if x_next.norm() >= trust_radius: # Exceeding trust region → project onto boundary
            tau = find_tau(x, d, trust_radius)
            return x + tau * d

        r_next = r - alpha * Hd

        if r_next.norm() < tol: # Converged
            return x_next

        beta = (r_next @ r_next) / (r @ r)
        d = r_next + beta * d

        x = x_next
        r = r_next

    return x

def solve_linear_psd(A, b, method=None):
    # Solve linear system Ax = b. We assume that A is symmetric and positive semi-definite

    if method is None:
        method = 'solve_ex'
        
    do_lstsq = False  # Fallback

    try:
        if method=='solve':
            x = torch.linalg.solve(A, b)

        elif method=='solve_ex':
            x = torch.linalg.solve_ex(A, b)[0]

        elif method == 'cholesky':
            L = torch.linalg.cholesky(A)
            x = torch.cholesky_solve(b.unsqueeze(-1), L, upper=False)
            
            return  x.squeeze()

        elif method == 'cholesky_ex':
            L = torch.linalg.cholesky_ex(A)[0]
            x = torch.cholesky_solve(b.unsqueeze(-1), L, upper=False)
            
            return  x.squeeze()

        elif method == 'QR':
            Q, R = torch.linalg.qr(A)
            x = torch.linalg.solve_triangular( R, (Q.T @ b).unsqueeze(1), upper=True).squeeze()

        elif method=='inv':
            x = torch.linalg.inv(A)@b

        elif method=='lstsq':
            do_lstsq = True

        else:
            assert False, f"Unknown method {method} in solve_linear_psd"

    except RuntimeError as e:
        if isinstance(e, NotImplementedError):
            raise        
        # If other methods fail (e.g., A is only PSD but not strictly positive definite),
        # fall back to a more robust method
        print(f"Warning: got error {str(e)} with method {method}, using lstsq instead")
        do_lstsq = True

    if do_lstsq:
        x = torch.linalg.lstsq(A, b).solution

    return x




def benchmark_linsolve(num_runs=10, printx=False):
    # This function is used to benchmark the linear solvers
    # It creates a random symmetric positive definite matrix and a random vector
    # Then it solves the linear system Ax = b using different methods
    # Finally, it prints the time taken for each method
    # It can be run as
    #   > python -c 'import utils; utils.benchmark_linsolve()'
    #
    # Arguments
    #   num_runs: Number of runs to average the time taken for each method
    #   printx: If True, print the solution for each method

    import time

    def get_A_b():
        n = 1000
        rand_mat = torch.randn(n, n)
        A = rand_mat @ rand_mat.T  # This creates a symmetric PSD matrix
        A += eye_like(A)*1e-4
        b = torch.randn(n)
        return A, b

    for method in ['solve','solve_ex','steihaug','cholesky','cholesky_ex','QR','lstsq','inv']:
        tot_time = 0
        if printx:
            print()
        for i in range(num_runs):
            torch.manual_seed(i)
            A, b = get_A_b()
            stime = time.time()
            if method != 'steihaug':
                x = solve_linear_psd(A, b, method=method)
            else:
                x = steihaug_toint_cg(A, b, trust_radius=100)
            tot_time += time.time() - stime
            empty_torch_cache()
            if printx:
                print(f'{method:15s} {x[:4].cpu().numpy()}')
        print(f'{method:15s} {tot_time:3f}')



