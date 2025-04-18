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
def minimize2(f, x0, max_iter=20, tol=None, tol_grad=None, method=None, callback=None, 
              line_search="strong_wolfe", lr=.1, lambda_ = 0.0):
    """
    Perform optimization using torch.optim.LBFGS with optional L2 regularization.
    
    Parameters:
        f           : objective function to minimize
        x0          : initial parameter tensor
        max_iter    : maximum number of iterations
        tol         : tolerance for stopping criterion
        method      : (unused) placeholder
        callback    : optional function called on each iteration
        line_search : line search strategy or None for none (default: "strong_wolfe")
        lambda_     : L2 regularization weight
        lr          : Learning rate (default 0.1)
        tol_grad    : Tolerance for first order optimality (gradient)
    """
    x = x0.clone().detach().requires_grad_(True)
    if tol is None: #  or tol >= 1e-6:
        tol = 1e-6
    if tol_grad is None:
        tol_grad = 1e-9
    lbfgs = torch.optim.LBFGS([x],
                    lr               = lr,
                    max_iter         = max_iter, 
                    history_size     = 1, 
                    tolerance_change = tol,
                    tolerance_grad   = tol_grad,
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