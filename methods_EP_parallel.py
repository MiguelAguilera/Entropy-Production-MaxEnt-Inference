import torch

# ###################################################################
# Optimization result container
# ###################################################################
class Result(object):
    """Simple container to store the result of the optimization."""
    def __init__(self, fun, x=None):
        self.fun = fun  # Final objective function value
        self.x   = x    # Final parameter values

# ###################################################################
# Custom minimization routine using PyTorch L-BFGS optimizer
# ###################################################################
def minimize2(f, x0, max_iter=20, tol=None, method=None, callback=None, 
              line_search="strong_wolfe", lambda_ = 0.0):
    """
    Perform optimization using torch.optim.LBFGS with optional L2 regularization.
    
    Parameters:
        f           : objective function to minimize
        x0          : initial parameter tensor
        max_iter    : maximum number of iterations
        tol         : tolerance for stopping criterion
        method      : (unused) placeholder
        callback    : optional function called on each iteration
        line_search : line search strategy (default: "strong_wolfe")
        lambda_     : L2 regularization weight
    """
    x = x0.clone().detach().requires_grad_(True)
    if tol >= 1e-6:
        tol = 1e-6
    lbfgs = torch.optim.LBFGS([x],
                    lr               = .1,
                    max_iter         = max_iter, 
                    history_size     = 1, 
                    tolerance_change = tol,
                    line_search_fn   = line_search
                    )

    # Define closure for LBFGS
    def closure():
        lbfgs.zero_grad()
        objective = f(x)
        
        # Add L2 regularization term
        reg_term = lambda_ * x.norm(p=2) ** 2
        loss = objective + reg_term  # Regularized loss

        loss.backward()

        if callback is not None:
            callback(x.data)

        return loss  # Return the regularized loss

    r = lbfgs.step(closure)
    cur_obj = f(x).item() #  recompute f(x) without regularization

    return Result(fun=cur_obj, x=x.detach().clone())

# ###################################################################
# Generate default arguments for the minimizer
# ###################################################################
def get_torchmin_args(S, tol_per_param=None):
    """
    Generate argument dictionary for torch minimization.
    
    Parameters:
        S              : input matrix
        tol_per_param  : tolerance per parameter (optional)
        
    Returns:
        Dictionary of arguments for `minimize2`
    """
    if tol_per_param is None:
        tol_per_param = 1e-4

    N = S.shape[0]
    return dict(x0       = torch.zeros(N * (N - 1) // 2, dtype=S.dtype),
                method   = 'l-bfgs',
                tol      = tol_per_param/N,
                max_iter = 50
                )

# ###################################################################
# Define MaxEnt objective function (for antisymmetric interaction matrix)
# ###################################################################
class MaxEntObjective(torch.nn.Module):
    def forward(self, theta, S, S1):
        """
        Compute maximum entropy objective function.
        
        Parameters:
            theta : upper-triangular parameters (N*(N-1)/2)
            S     : sample matrix (N x reps)
            S1    : auxiliary matrix (same shape as S)

        Returns:
            Scalar objective value
        """
        N, rep = S.shape
        th = torch.zeros((N, N), dtype=theta.dtype)
        triu_indices = torch.triu_indices(N, N, offset=1)
        th[triu_indices[0], triu_indices[1]] = theta
        th1= th - th.T
        thS = (th1 - th1.T) @ S
        thf_odd =torch.sum(S1 * thS, axis=0) 
        thf_min = torch.min(thf_odd)
        sig = torch.mean(thf_odd) + thf_min - torch.log(torch.mean(torch.exp(-thf_odd + thf_min)))
        return sig

# Create a global instance of the objective function
obj_fn = MaxEntObjective()

# ###################################################################
# Run maximum entropy optimization
# ###################################################################
def get_torch(S, S1, max_iter=None, tol_per_param=None, mode=2, lambda_=0.0):
    """
    Wrapper to perform max-entropy optimization given data matrices S and S1.
    
    Parameters:
        S             : data matrix (N x reps)
        S1            : auxiliary matrix
        max_iter      : optional max iterations
        tol_per_param : optional tolerance per parameter
        mode          : unused
        lambda_       : regularization weight
    """
    N = int(S.shape[0])

    # Define negative objective for minimization
    f = lambda theta: -obj_fn(theta, S, S1)

    # Get default minimizer arguments
    args = get_torchmin_args(S, tol_per_param)
    args['lambda_'] = lambda_

    # Run optimization
    res = minimize2(f, **args)

    print('     max_theta', torch.max(torch.abs(res.x)))
    return -res.fun  # Return original (positive) objective value

