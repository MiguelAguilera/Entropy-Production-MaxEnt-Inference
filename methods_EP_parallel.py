import torch
from optimize import *

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
        
        # Load upper triange
        th = torch.zeros((N, N), dtype=theta.dtype)
        triu_indices = torch.triu_indices(N, N, offset=1)
        th[triu_indices[0], triu_indices[1]] = theta
        
        # Compute the product of antisymmetric couplings by sample matrix
        thS = (th - th.T) @ S
        
        # Element-wise multiply with S1 and sum over rows (axis=0) to get one value per sample
        thf_odd =torch.sum(S1 * thS, axis=0) 
        
        # Compute the log-sum-exp trick to stabilize exponential averaging
        thf_min = torch.min(thf_odd)
        sig = torch.mean(thf_odd) + thf_min - torch.log(torch.mean(torch.exp(-thf_odd + thf_min)))
        
        return sig/N        # Scale by N to normalize for tolerance calculation

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
    return -res.fun*N  # Return original (positive) objective value

