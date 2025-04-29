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


def eye_like(A):
    return torch.eye(A.size(-1), dtype=A.dtype, device=A.device)


def steihaug_toint_cg(A, b, trust_radius, tol=1e-10, max_iter=250):
    def find_tau(x, d, trust_radius):
        # Solve ||x + tau*d|| = trust_radius for tau
        a = d @ d
        b = 2 * (x @ d)
        c = (x @ x) - trust_radius**2
        sqrt_discriminant = torch.sqrt(b**2 - 4*a*c)
        tau = (-b + sqrt_discriminant) / (2*a)
        return tau

    x = torch.zeros_like(b)
    r = b.clone()
    d = r

    for i in range(max_iter):
        Hd = A @ d
        dHd = d @ Hd

        if dHd <= 0:
            # Negative curvature, move to boundary
            tau = find_tau(x, d, trust_radius)
            return x + tau * d

        alpha = (r @ r) / dHd
        x_new = x + alpha * d

        if x_new.norm() >= trust_radius:
            tau = find_tau(x, d, trust_radius)
            return x + tau * d

        r_new = r + alpha * Hd

        if r_new.norm() < tol:
            return x_new

        beta = (r_new @ r_new) / (r @ r)
        d = -r_new + beta * d
        x = x_new
        r = r_new

    return x


def solve_linear_psd(A, b, method=None, eps=0):
    # Solve linear system Ax = b. We assume that A is symmetric and positive semi-definite
    assert not is_infnan(b.sum())
    assert not is_infnan(A.sum())

    if method is None:
        method = 'solve_ex'
        

    do_lstsq = False
    A2 = A #if eps == 0 else A + eye_like(A)*eps

    try:
        if method=='solve':
            x = torch.linalg.solve(A2, b)

        elif method == 'cg':
            x = steihaug_toint_cg(A2, b, trust_radius=10)

        elif method=='solve_ex':
            x = torch.linalg.solve_ex(A2, b)[0]

        elif method == 'cholesky':
            L = torch.linalg.cholesky(A2)
            x = torch.cholesky_solve(b.unsqueeze(-1), L, upper=False)
            
            return  x.squeeze()

        elif method == 'cholesky_ex':
            L = torch.linalg.cholesky_ex(A2)[0]
            x = torch.cholesky_solve(b.unsqueeze(-1), L, upper=False)
            
            return  x.squeeze()

        elif method == 'QR':
            Q, R = torch.linalg.qr(A2)
            x = torch.linalg.solve_triangular( R, (Q.T @ b).unsqueeze(1), upper=True).squeeze()

        elif method=='inv':
            x = torch.linalg.inv(A2)@b

        elif method=='lstsq':
            do_lstsq = True

        #elif method=='CG':
        #    import linalg 
        #    x, _ = linalg.CG(A2, b, max_iter=10)

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
        x = torch.linalg.lstsq(A2, b).solution

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


