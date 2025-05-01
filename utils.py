import torch
import numpy as np

def set_default_device():
    """
    Determines the best available device for PyTorch operations and sets it as default.
    Returns:
        torch.device: The device that was set as default ('mps', 'cuda', or 'cpu')
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # Set MPS as default device
        torch.set_default_device(device)
        import warnings
        warnings.filterwarnings("ignore", message="The operator 'aten::_linalg_solve_ex.result' is not currently supported on the MPS backend and will fall back to run on the CPU", category=UserWarning)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        # Set CUDA as default device
        torch.set_default_device(device)
    else:
        device = torch.device("cpu")
        # CPU is already the default, but we can set it explicitly
        torch.set_default_device(device)
    return device


def empty_cache():  # Empty torch cache
    if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
        torch.cuda.empty_cache()
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def is_float_type(obj):
    # Check if it's a Python built-in float
    if isinstance(obj, float):
        return "Python built-in float"
    
    # Check if it's a NumPy float
    try:
        import numpy as np
        if isinstance(obj, np.floating):
            return "NumPy float"
    except ImportError:
        pass
    
    # Check if it's a PyTorch float
    try:
        import torch
        if isinstance(obj, torch.Tensor) and obj.dtype.is_floating_point:
            return "PyTorch float tensor"
    except ImportError:
        pass
    
    # If none of the above
    return False


def eye_like(A):
    return torch.eye(A.size(-1), dtype=A.dtype, device=A.device)


def steihaug_toint_cg(A, b, trust_radius, tol=1e-10, max_iter=None):
    """
    Steihaug-Toint Conjugate Gradient method for approximately solving
    min_x 0.5 x^T A x - b^T x  subject to ||x|| <= trust_radius
    where A is symmetric (and possibly only positive semi-definite).
    
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
    max_iter = max_iter or 2 * n
    x = torch.zeros_like(b)
    r = b.clone()
    d = r.clone()

    if r.norm() < tol:
        return x

    for k in range(max_iter):
        Hd = A @ d
        dHd = d @ Hd

        if dHd <= 0:
            # Negative curvature detected → move to boundary
            tau = find_tau(x, d, trust_radius)
            return x + tau * d

        alpha = (r @ r) / dHd
        x_next = x + alpha * d

        if x_next.norm() >= trust_radius:
            # Exceeding trust region → project onto boundary
            tau = find_tau(x, d, trust_radius)
            return x + tau * d

        r_next = r - alpha * Hd

        if r_next.norm() < tol:
            # Converged
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





# ===================================
# Helpful tensor processing functions
# ===================================

def remove_i_rowcol(X, i):
    # Remove the i-th row and column from matrix Ks.
    X1 = torch.cat([X [:i, :], X [i+1:, :]], dim=0)
    X2 = torch.cat([X1[:, :i], X1[:, i+1:]], dim=1)
    return X2

def remove_i(A, i):
    # Remove the i-th element from a 1D tensor A.
    r = torch.cat((A[:i], A[i+1:]))
    return r

def add_i(x,i):
    # Insert 0 in position i
    return torch.cat((x[:i], torch.tensor([0.0], device=x.device), x[i:]))   

def add_i_rowcol(X,i):
    # Insert 0 row and column
    d=X.device
    n=X.shape[0]
    X1 = torch.cat((X[:i,:], torch.zeros((1,n), device=d), X[i:,:]),dim=0)   
    X2 = torch.cat((X1[:,:i], torch.zeros((n+1,1), device=d), X1[:,i:]),dim=1)
    return X2   


# Miscellaneous

def is_infnan(x):
    x = float(x)
    return np.isinf(x) or np.isnan(x)


