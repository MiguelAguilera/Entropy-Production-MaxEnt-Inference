import os, time
from collections import namedtuple
from collections.abc import Iterable
import numpy as np

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]='1'
import torch

from utils import numpy_to_torch, is_infnan, eye_like
import linear_solvers

# This is the Objective class that should be implemented by the user, and it should 
# provide the methods get_objective, get_gradient, and get_hessian (if needed by the optimizer). 
# Instances can be passed in as optimize(..., optimizer=objective_instance) 
class Objective(object):
    def get_objective(self, theta): # Return objective value for parameters theta
        raise NotImplementedError
    def get_gradient(self, theta): # Return gradient of the objective function for parameters theta
        raise NotImplementedError 
    def get_hessian(self, theta):  # Return hessian of the objective function for parameters theta
        raise NotImplementedError 
    


# This is the base class for all optimizers. It provides a common interface for all optimizers
# and defines the methods that should be implemented by each specific optimizer.
class Optimizer(object):
    minimize = True

    def reset_state(self): # Reset the optimizer internal state if needed
        pass

    def __init__(self, max_iter=1000, patience=10, verbose=0):
        # Initialize optimizer with universally-shared parameters
        # Arguments:
        #   minimize (bool) : whether to minimize or maximize the objective function
        #   max_iter (int)  : maximum number of iterations
        #   patience (int)  : number of iterations to wait for validation improvement before stopping
        #   verbose (int)   : verbosity level
        self.max_iter = max_iter
        self.patience = patience
        self.verbose = verbose

        self.reset_state()  # Reset the optimizer state


    def set_minimize_flag(self, minimize):
        # Set the flag for minimization or maximization
        self.minimize = minimize


    def msg(self, s, t=None):
        print(f"{self.__class__.__name__} : " + (f"iter={t:4d} :" if t is not None else "") + s)

    def get_update(self, t, objective, x):
        # This method provides the specific update logic for each optimizer, returning 
        # the updated parameters x_t+1, based on the current iteration t, objective object, 
        # and current parameters x
        #
        # IMPORTANT: this method can return either
        #   1. new_x (the updated parameters)
        #   2. (new_x, f_new_trn) (the updated parameters and the new training objective)
        # The logic in optimize(...) will handle both cases
        raise NotImplementedError


class GradientDescent(Optimizer):
    def __init__(self, lr=0.001, max_iter=10000, **kwargs):
        # Gradient ascent optimizer. lr is the learning rate
        self.lr = lr
        super().__init__(max_iter=max_iter, **kwargs)

    def get_update(self, t, objective, x):
        grad = objective.get_gradient(x)
        if self.minimize: grad = -grad
        return x + self.lr * grad
    

class GradientDescentBB(GradientDescent):
    # Gradient ascent with Barzilai-Borwein step sizes (short-step version)
    def __init__(self, lr=0.001, max_iter=1000, **kwargs):
        # TODO: check if optimizer can be unstable if lr is too large (and maybe even if too low), 
        # this can be tested using example_general.py
        # lr is the initial learning rate
        super().__init__(max_iter=max_iter, lr=lr, **kwargs)

    def reset_state(self):
        # Reset the optimizer state
        self.previous_grad = None
        self.previous_x    = None

    def get_update(self, t, objective, x):
        grad = objective.get_gradient(x)
        if self.previous_grad is not None and self.previous_x is not None:
            last_delta_x = x    - self.previous_x
            d_grad       = grad - self.previous_grad
            
            # BB short step size
            alpha        = (last_delta_x @ d_grad) / (d_grad @ d_grad)

            if is_infnan(alpha):
                # If alpha is NaN, fall back to the default learning rate
                if self.verbose: print(f'GradientDescentBB : Invalid alpha!')
                alpha = self.lr
        else:
            alpha        = self.lr

        self.previous_grad = grad
        self.previous_x    = x

        # Like with Newton's method, BB update is actually the same for minimization and maximization
        return x - alpha * grad  


class Adam(GradientDescent):
    # Gradient-based optimized using Adam (Adaptive Moment Estimation) optimizer by Kingma and Ba
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, skip_warm_up=False, eps=1e-8, **kwargs):
        # Adam optimizer.
        # Arguments:
        #   lr (float) : learning rate
        #   max_iter (int) : maximum number of iterations
        #   beta1 (float) : exponential decay rate for the first moment estimates
        #   beta2 (float) : exponential decay rate for the second moment estimates
        #   skip_warm_up (bool) : if True, we skip the warm-up phase for the first moment estimates
        #   eps (float) : small constant to prevent division by zero
        #   verbose (int) : verbosity level
        self.beta1 = beta1
        self.beta2 = beta2
        self.skip_warm_up = skip_warm_up
        self.eps = eps
        super().__init__(lr=lr, **kwargs)


    def reset_state(self): # Reset the optimizer state
        self.m = None
        self.v = None

    def get_update(self, t, objective, x):
        if self.m is None or self.v is None:
            self.m = torch.zeros_like(x)
            self.v = torch.zeros_like(x)

        grad = objective.get_gradient(x)
        if self.minimize: grad = -grad

        # Adam moment updates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        if self.skip_warm_up:
            m_hat = self.m
            v_hat = self.v
        else:
            m_hat = self.m / (1 - self.beta1 ** (t+1))
            v_hat = self.v / (1 - self.beta2 ** (t+1))

        # Compute parameter update
        delta_x = self.lr * m_hat / (v_hat.sqrt() + self.eps)
        return x + delta_x
    

class NewtonMethod(Optimizer):
    # Newton-Raphson method optimizer
    def __init__(self, linsolve_eps=1e-8, max_iter=100, **kwargs):
        self.linsolve_eps = linsolve_eps   # regularization parameter for linear solvers (helps numerical stability)
        super().__init__(max_iter=max_iter, **kwargs)

    def get_regularized_hessian(self, objective, x):
        # Get the Hessian matrix and add a small regularization term
        H  = objective.get_hessian(x)
        if is_infnan(H.sum()):   # Error occured, usually it means x is too big
            if self.verbose: print(f'NewtonMethod : [Stopping] Invalid Hessian')
            return None 
        return H + self.linsolve_eps * eye_like(H)

    def get_update(self, t, objective, x):
        grad = objective.get_gradient(x)
            
        H_reg = self.get_regularized_hessian(objective, x)
        if H_reg is None: return None 

        return x - linear_solvers.solve_linear_psd(A=H_reg, b=grad) 



class NewtonMethodTrustRegion(NewtonMethod):
    # Newton's method with trust region constraint
    def __init__(self, max_iter=1000, trust_radius=1, **kwargs):
        self.trust_radius = trust_radius  # maximum trust region
        super().__init__(max_iter=max_iter, **kwargs)

    def get_update(self, t, objective, x):
        grad = objective.get_gradient(x)
        H_reg = self.get_regularized_hessian(objective, x)
        if H_reg is None: return None 

        if not self.minimize:
            grad, H_reg = -grad, -H_reg

        # Solve the constrained trust region problem using the
        # Steihaug-Toint Truncated Conjugate-Gradient Method
        delta_x = linear_solvers.steihaug_toint_cg(A=H_reg, b=grad, trust_radius=self.trust_radius)
        
        return x - delta_x

        
class TRON(NewtonMethodTrustRegion):
    # TRON method, Newton's method with adaptive trust region constraints
    # Reference:
    #   Lin and Moré, Newton's method for large bound-constrained optimization problems
    #   SIAM Journal on Optimization 9 (4), 1100-1127
    def __init__(self, max_iter=1000, trust_radius=1, eta0=0.0, eta1=0.25, eta2=0.75, 
                    trust_radius_min=1e-3, trust_radius_max=1000.0,
                    trust_radius_adjust_max_iter=100, linsolve_eps=1e-8, **kwargs):
        # Parameters
        #   linsolve_eps     : regularization parameter for linear solvers (helps numerical stability)
        # Trust-region-related arguments (all optional):
        #   trust_radius (float or None) : maximum trust region (if None, trust region constraint are not used)
        #   solve_constrained (bool)     : if True, we find direction by solving the constrained trust-region problem 
        #                                  if False, we simply rescale usual Newton step to desired maximum norm 
        #   adjust_radius (bool)         : change trust-region radius in an adaptive way
        #   eta0=0.0, eta1=0.25, eta2=0.75,trust_radius_min,trust_radius_max,trust_radius_adjust_max_iter
        #                                : hyperparameters for adjusting trust radius
        self.linsolve_eps = linsolve_eps    # regularization parameter for linear solvers (helps numerical stability)
        self.trust_radius = trust_radius    # initial trust region radius
        self.trust_radius_min = trust_radius_min  # minimum trust region radius
        self.trust_radius_max = trust_radius_max  # maximum trust region radius
        self.trust_radius_adjust_max_iter = trust_radius_adjust_max_iter  # maximum number of iterations for adjusting trust radius
        self.eta0 = eta0 
        self.eta1 = eta1
        self.eta2 = eta2

        super().__init__(max_iter=max_iter, **kwargs)


    def reset_state(self):
        self.adjusted_trust_radius = float(self.trust_radius)
        self.f_last_trn = None   # keep track of the last training objective


    def get_update(self, t, objective, x):
        H_reg = self.get_regularized_hessian(objective, x)
        if H_reg is None: return None 

        grad = objective.get_gradient(x)
        if not self.minimize:
            grad, H_reg = -grad, -H_reg

        if not hasattr(self, 'f_last_trn') or self.f_last_trn is None:
            self.f_last_trn = objective.get_objective(x)
            if is_infnan(self.f_last_trn):
                if self.verbose: self.msg(f"[Stopping] Invalid training objective {self.f_last_trn}", t)
                return None
            
        for tr_iter in range(self.trust_radius_adjust_max_iter): # loop until adjusted_trust_radius is adjusted properly
            delta_x    = linear_solvers.steihaug_toint_cg(A=H_reg, b=grad, trust_radius=self.adjusted_trust_radius)
            new_x      = x - delta_x
            f_new_trn  = objective.get_objective(new_x)

            pred_improvement = (grad @ delta_x + 0.5 * delta_x @ (H_reg @ delta_x))
            improvement      = f_new_trn - self.f_last_trn
            if self.minimize: 
                improvement      = -improvement
            if self.verbose > 1: self.msg(f"pred_improvement={pred_improvement:.3f} improvement={improvement:.3f}", t)

            rho = improvement / (pred_improvement + 1e-20)

            assert not is_infnan(rho), "rho is not a valid number in adjust_radius code. Try disabling adjust_radius=False"
            if rho > self.eta0:      # accept new parameters
                break

            if self.adjusted_trust_radius < self.trust_radius_min:
                if self.verbose: self.msg(f"trust_radius_min={self.trust_radius_min} reached!", t)
                self.adjusted_trust_radius = self.trust_radius_min
                last_round = True
                break

            if rho < self.eta1:
                self.adjusted_trust_radius *= 0.25
                if self.verbose > 1: self.msg(f"reducing adjusted_trust_radius to {self.adjusted_trust_radius} in trust radius iteration={tr_iter}", t)

            elif rho > self.eta2 and delta_x.norm() >= self.adjusted_trust_radius:
                self.adjusted_trust_radius = min(2.0 * self.adjusted_trust_radius, self.trust_radius_max)
                if self.verbose > 1: self.msg(f"increasing adjusted_trust_radius to {self.adjusted_trust_radius} in trust radius iteration={tr_iter}", t)


        else: # for loop finished without breaking
            if self.verbose: self.msg(f"max_iter reached in adjusted_trust_radius loop!", t)


        self.f_last_trn = f_new_trn
        return new_x, f_new_trn


OPTIMIZERS = [GradientDescent, GradientDescentBB, Adam,
             NewtonMethod, NewtonMethodTrustRegion, TRON]


# This is the solution object that is returned by the optimize function
Solution = namedtuple('Solution', ['objective', 'x', 'val_objective'], defaults=[None])

def optimize(x0, objective, minimize=True, validation=None, optimizer=None, 
             tol=1e-8, max_trn_objective=None, max_val_objective=None, min_trn_objective=None, min_val_objective=None,
             verbose=0, report_every=10, skip_max_iter_warning=False, device=None
             ):
    # Optimize function using the specified optimizer
    # Arguments:
    #   x0               : torch tensor specifiying initial parameters (set to 0s if None)
    #   objective        : Instance of Objective class that provides get_objective, get_gradient and/or get_hessian information
    #   minimize         : whether to minimize or maximize the objective function
    #   validation       : if not None, we use this Objective instance for early stopping. It should provide get_objective method
    #   test             : if not None, we use this Objective instance for evaluating the objective. It should provide get_objective method
    #   tol (float)      : early stopping once objective does not improve by more than tol
    #   max_trn_objective (float) : maximum training objective value, clip and stop if exceeded
    #   max_val_objective (float) : maximum validation objective value, clip and stop if exceeded
    #   min_trn_objective (float) : minimum training objective value, clip and stop if below
    #   min_val_objective (float) : minimum validation objective value, clip and stop if below
    #   optimizer (str)  : Instance of Optimizer class to use. Should support .reset_state() and .get_update() methods
    #                      If None, we use the default optimizer (GradientDescentBB)
    #   verbose (int)    : Verbosity level for printing debugging information (0=no printing)
    #   report_every (int) : if verbose > 1, we report every report_every iterations
    #   skip_max_iter_warning (bool) : if True, we skip the warning about max_iter being reached
    #
    # Returns:
    #   Solution object with the following attributes:
    #       objective     : final objective value on training data set
    #       x             : final parameters, as a torch tensor
    #       val_objective : final objective value on validation data set (if validation is not None)

    if optimizer is None: 
        optimizer = GradientDescentBB(verbose=verbose)
    else:
        optimizer.reset_state()
    optimizer.set_minimize_flag(minimize)
    
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    

    with torch.no_grad():  # We calculate our own gradients, so we don't need to torch to do it (sometimes a bit faster)
        x = numpy_to_torch(x0).to(device)

        f_cur_trn = f_new_trn = np.nan
        if validation is not None:
            f_new_val = f_best_val = np.nan
            best_val_x       = x.clone()
            best_val_iter    = 0
            patience_counter = 0
        
        old_time         = time.time()

        for t in range(optimizer.max_iter):
            r = optimizer.get_update(t=t, objective=objective, x=x)
            
            # get_update can return either new_x or (new_x, f_new_trn)
            if isinstance(r, Iterable) and len(r) == 2:  
                new_x, f_new_trn = r
            else:
                new_x = r
                if new_x is not None:
                    f_new_trn  = objective.get_objective(x) 

            if new_x is None:
                if verbose: optimizer.msg(f"[Stopping] Invalid update, x={x}", t)
                break

            if is_infnan(f_new_trn):
                if verbose: optimizer.msg(f"[Stopping] Invalid training objective {f_new_trn}", t)
                break

            if verbose and verbose > 1 and report_every > 0 and t % report_every == 0:
                new_time = time.time()
                optimizer.msg(f'{(new_time - old_time)/report_every:4.2f}s/iter | f_cur_trn={f_cur_trn: 5.3f}' + 
                    (f' f_new_val={f_new_val: 5.3f} f_best_val={f_best_val: 5.3f} patience_counter={patience_counter}' if validation is not None else '') , t)
                old_time = new_time

            if validation is not None:
                f_new_val = validation.get_objective(new_x) 
                if is_infnan(f_new_val):
                    if verbose: optimizer.msg(f"[Stopping] Invalid validation objective {f_new_val}", t)
                    break

            x = new_x

            if max_trn_objective is not None and f_new_trn > max_trn_objective:
                f_cur_trn = max_trn_objective
                if verbose: optimizer.msg(f"[Clipping] f_new_trn > max_trn_objective, ‖x‖∞={torch.abs(x).max():3.2f}", t)
                break 

            if min_trn_objective is not None and f_new_trn < min_trn_objective:
                f_cur_trn = min_trn_objective
                if verbose: optimizer.msg(f"[Clipping] f_new_trn < min_trn_objective, ‖x‖∞={torch.abs(x).max():3.2f}", t)
                break 

            if abs(f_new_trn - f_cur_trn) < tol:
                f_cur_trn = f_new_trn
                if verbose: optimizer.msg(f"[Converged] Training objective change below tol={tol}", t)
                break 

            f_cur_trn = f_new_trn

            if validation is not None: 
                if max_val_objective is not None and f_new_val > max_val_objective:
                    f_best_val = max_val_objective
                    best_val_x = x
                    if verbose: optimizer.msg(f"[Clipping] f_new_val > max_val_objective, ‖x‖∞={torch.abs(x).max():3.2f}", t)
                    break

                if min_val_objective is not None and f_new_val < min_val_objective:
                    f_best_val = min_val_objective
                    best_val_x = x
                    if verbose: optimizer.msg(f"[Clipping] f_new_val < min_val_objective, ‖x‖∞={torch.abs(x).max():3.2f}", t)
                    break

                improved = is_infnan(f_best_val) or (f_new_val < f_best_val if minimize else f_new_val > f_best_val)
                if improved:
                    if verbose > 1 and t-best_val_iter>1: 
                        optimizer.msg(f"[Patience] Resetting patience counter, last improvement {t-best_val_iter} steps ago", t)
                    f_best_val       = f_new_val
                    best_val_x       = x.clone()
                    patience_counter = 0
                    best_val_iter    = t

                elif patience_counter >= optimizer.patience:
                    if verbose:
                        optimizer.msg(f"[Stopping] Validation objective did not improve for {optimizer.patience} steps (last improvement {t-best_val_iter} steps ago)", t)
                    break
                
                else:
                    patience_counter += 1
            
                    
        else:   # for loop did not break
            if not skip_max_iter_warning or verbose:
                optimizer.msg(f'max_iter {optimizer.max_iter} reached before convergence.' + 
                              ('May want to increase max_iter' if not skip_max_iter_warning else ''))

        if validation is not None:
            return Solution(objective=f_cur_trn, x=best_val_x, val_objective=f_best_val)
        else:
            return Solution(objective=f_cur_trn, x=x, val_objective=None)



