import os, time
from collections import namedtuple
from collections.abc import Iterable
import numpy as np

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]='1'
import torch

from utils import numpy_to_torch, is_infnan, solve_linear_psd, steihaug_toint_cg

## TODO: Call Objective not Dataset
##  TODO: cache mean?
## UPDATE: can also return new x and f_new_trn
class Optimizer(object):
    def msg(self, s, t=None):
        print(f"{self.__class__.__name__} : " + (f"iter={t:4d} :" if t is not None else "") + s)

    def get_update(self, t, objective, x):
        # This method should be implemented in subclasses of Optimizer
        # to provide the specific update logic for each optimizer.
        # It should return the updated parameters x_t+1
        # based on the current iteration t, objective object, and current parameters x.
        raise NotImplementedError("get_update method not implemented")
    

class GradientDescent(Optimizer):
    def __init__(self, lr=0.001, minimize=True, verbose=0):
        # Gradient ascent optimizer. lr is the learning rate
        self.lr = lr
        self.minimize = minimize  # True for minimization, False for maximization
        self.verbose = verbose

    def get_update(self, t, objective, x):
        grad = objective.get_gradient(x)
        if self.minimize:
            return x - self.lr * grad
        else:
            return x + self.lr * grad
    

class GradientDescentBB(Optimizer):
    # Gradient ascent with Barzilai-Borwein step sizes (short-step version)
    def __init__(self, lr=0.001, minimize=True, verbose=0):
        self.lr = lr  # initial learning rate
        self.previous_grad = None
        self.previous_x    = None
        self.verbose = verbose

    def get_update(self, t, objective, x):
        grad = objective.get_gradient(x)
        if self.previous_grad is not None and self.previous_x is not None:
            last_delta_x = x    - self.previous_x
            d_grad       = grad - self.previous_grad
            
            # BB short step size. Abs takes care of both minimization and maximization
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


class Adam(Optimizer):
    def __init__(self, lr=0.001, minimize=True, verbose=0, beta1=0.9, beta2=0.999, skip_warm_up=False, eps=1e-8):
        # Adam optimizer
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.skip_warm_up = skip_warm_up
        self.eps = eps

        self.minimize = minimize  # True for minimization, False for maximization
        self.verbose = verbose

    def get_update(self, t, objective, x):

        if not hasattr(self, 'm'):
            self.m = torch.zeros_like(x)
            self.v = torch.zeros_like(x)

        grad = objective.get_gradient(x)

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
        return x + (-delta_x if self.minimize else delta_x)
    

class NewtonMethod(Optimizer):
    def __init__(self, minimize=True, verbose=0, linsolve_eps=1e-8):
        # Newton method optimizer
        self.linsolve_eps = linsolve_eps
        self.verbose = verbose

    def get_regularized_hessian(self, objective, x):
        H  = objective.get_hessian(x)
        if is_infnan(H.sum()):   # Error occured, usually it means x is too big
            if self.verbose: print(f'NewtonMethod : [Stopping] Invalid Hessian')
            return None 
        return H + self.linsolve_eps * torch.eye(len(x), device=x.device)
    

    def get_update(self, t, objective, x):
        grad = objective.get_gradient(x)
            
        H_reg = self.get_regularized_hessian(objective, x)
        if H_reg is None: return None 

        return x - solve_linear_psd(A=H_reg, b=grad) 



class NewtonMethodTrustRegion(NewtonMethod):
    # Newton's method with adaptive trust region constraints
    def __init__(self, trust_radius=1, minimize=True, verbose=0, linsolve_eps=1e-8):
        self.linsolve_eps = linsolve_eps
        self.trust_radius = trust_radius
        self.minimize     = minimize
        self.verbose      = verbose

    def get_delta_x(self, H, grad):
        if self.minimize:
            return steihaug_toint_cg(A=H, b=grad, trust_radius=self.trust_radius)
        else:
            return steihaug_toint_cg(A=-H, b=-grad, trust_radius=self.trust_radius)
    def get_update(self, t, objective, x, verbose=0):
        grad = objective.get_gradient(x)
            
        H_reg = self.get_regularized_hessian(objective, x)
        if H_reg is None: return None 
        
        # Solve the constrained trust region problem using the
        # Steihaug-Toint Truncated Conjugate-Gradient Method
        return x - self.get_delta_x(H_reg, grad)

        
class TRON(NewtonMethodTrustRegion):
    # TRON method, Newton's method with adaptive trust region constraints
    # Reference:
    #   Newton's method for large bound-constrained optimization problems
    #   CJ Lin, JJ Moré
    #   SIAM Journal on Optimization 9 (4), 1100-1127
    #
    def __init__(self, trust_radius=1, eta0=0.0, eta1=0.25, eta2=0.75, trust_radius_min=1e-3, trust_radius_max=1000.0,
                    trust_radius_adjust_max_iter=100, minimize=True, verbose=0, linsolve_eps=1e-8):
        # Parameters
        #   linsolve_eps     : regularization parameter for linear solvers (helps numerical stability)
        # Trust-region-related arguments (all optional):
        #   trust_radius (float or None) : maximum trust region (if None, trust region constraint are not used)
        #   solve_constrained (bool)     : if True, we find direction by solving the constrained trust-region problem 
        #                                  if False, we simply rescale usual Newton step to desired maximum norm 
        #   adjust_radius (bool)         : change trust-region radius in an adaptive way
        #   eta0=0.0, eta1=0.25, eta2=0.75,trust_radius_min,trust_radius_max,trust_radius_adjust_max_iter
        #                                : hyperparameters for adjusting trust radius
        self.linsolve_eps = linsolve_eps
        self.verbose = verbose
        self.trust_radius = trust_radius
        self.eta0 = eta0
        self.eta1 = eta1
        self.eta2 = eta2
        self.trust_radius_min = trust_radius_min
        self.trust_radius_max = trust_radius_max
        self.trust_radius_adjust_max_iter = trust_radius_adjust_max_iter
        self.trust_radius = trust_radius

        self.minimize = minimize  

    def get_update(self, t, objective, x, verbose=0):
        grad = objective.get_gradient(x)
            
        H_reg = self.get_regularized_hessian(objective, x)
        if H_reg is None: return None 


        if not hasattr(self, 'f_last_trn'):
            self.f_last_trn = objective.get_objective(x)
            if is_infnan(self.f_last_trn):
                if verbose: self.msg(f"[Stopping] Invalid training objective {self.f_last_trn}", t)
                return None
            
        
        for tr_iter in range(self.trust_radius_adjust_max_iter): # loop until trust_radius is adjusted properly
            delta_x    = self.get_delta_x(H_reg, grad)
            new_x      = x - delta_x
            f_new_trn  = objective.get_objective(new_x)

            pred_improvement = -(grad @ delta_x + 0.5 * delta_x @ (H_reg @ delta_x))
            improvement      = f_new_trn - self.f_last_trn
            if self.minimize: 
                improvement      = -improvement
                pred_improvement = -pred_improvement
                pass

            rho = improvement / (pred_improvement + 1e-20)

            assert not is_infnan(rho), "rho is not a valid number in adjust_radius code. Try disabling adjust_radius=False"
            if rho > self.eta0:      # accept new parameters
                break

            if self.trust_radius < self.trust_radius_min:
                if verbose: self.msg(f"trust_radius_min={self.trust_radius_min} reached!", t)
                self.trust_radius = self.trust_radius_min
                last_round = True
                break

            if rho < self.eta1:
                self.trust_radius *= 0.25
                if verbose > 1: self.msg(f"reducing trust_radius to {self.trust_radius} in trust radius iteration={tr_iter}", t)

            elif rho > self.eta2 and delta_x.norm() >= self.trust_radius:
                self.trust_radius = min(2.0 * self.trust_radius, self.trust_radius_max)
                if verbose > 1: self.msg(f"increasing trust_radius to {self.trust_radius} in trust radius iteration={tr_iter}", t)


        else: # for loop finished without breaking
            if verbose: self.msg(f"max_iter reached in adjust_radius loop!", t)


        self.f_last_trn = f_new_trn
        return new_x, f_new_trn




OPTIMIZERS = {
    'GradientDescent': GradientDescent,
    'GradientDescentBB': GradientDescentBB,
    'Adam': Adam,
    'NewtonMethod': NewtonMethod,
    'NewtonMethodTrustRegion': NewtonMethodTrustRegion,
    'TRON': TRON,}

Solution = namedtuple('Solution', ['objective', 'x', 'trn_objective'], defaults=[None])



def optimize(x0, objective, minimize=True, optimizer='GradientDescentBB', optimizer_args=None, 
             max_iter=None,  tol=1e-8, max_trn_objective=None, max_val_objective=None, min_trn_objective=None, min_val_objective=None,
             validation_objective=None, test_objective=None, patience=None, 
             verbose=0, report_every=10,
             ):
    # Optimize function using the specified optimizer
    # Arguments:
    #   x0               : torch tensor specifiying initial parameters (set to 0s if None)
    #   objective        : Object providing objective, gradient and/or Hessian information
    #   minimize         : whether to minimize or maximize the objective function
    #   optimizer (str)  : optimizer to use (see list above)
    #   optimizer_args (dict) : arguments to pass to the optimizer
    #   max_iter (int)   : maximum number of iterations
    #   tol (float)      : early stopping once objective does not improve by more than tol
    #   max_trn_objective (float) : maximum training objective value, clip and stop if exceeded
    #   max_val_objective (float) : maximum validation objective value, clip and stop if exceeded
    #   min_trn_objective (float) : minimum training objective value, clip and stop if below
    #   min_val_objective (float) : minimum validation objective value, clip and stop if below
    #   validation_objective : if not None, we use this dataset for early stopping. It should provide get_objective method
    #   test_objective   : if not None, we use this dataset for evaluating the objective. It should provide get_objective method
    #   patience (int)   : number of iterations to wait for validation improvement before stopping
    #   verbose (int)    : Verbosity level for printing debugging information (0=no printing)
    #   report_every (int) : if verbose > 1, we report every report_every iterations


    if max_iter is None:
        max_iter = 10000

    if patience is None:
        patience = 10

    if optimizer_args is None:
        optimizer_args = {}

    with torch.no_grad():  # We calculate our own gradients, so we don't need to torch to do it (sometimes a bit faster)

        # # Empty datasets
        # if objective.nsamples == 0 or \
        #     (validation_objective is not None and validation_objective.nsamples == 0) or \
        #     (test_objective       is not None and test_objective.nsamples       == 0):  
        #     if verbose: print(f"optimize found no samples in dataset")
        #     return Solution(objective=0.0, x=torch.zeros(objective.nobservables, device=objective.device), trn_objective=0.0)
        
        x = numpy_to_torch(x0)

        optimizer = OPTIMIZERS[optimizer](minimize=minimize, verbose=verbose, **optimizer_args)

        f_new_trn = np.nan
        if validation_objective is not None:
            f_new_val = f_best_val = np.nan
            best_val_x       = x.clone()
            best_val_iter    = 0
            patience_counter = 0
        
        old_time         = time.time()

        for t in range(max_iter):

            f_cur_trn = f_new_trn

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
                    (f' f_new_val={f_new_val: 5.3f} f_best_val={f_best_val: 5.3f} patience_counter={patience_counter}' if validation_objective is not None else '') , t)
                old_time = new_time

            if validation_objective is not None:
                f_new_val = validation_objective.get_objective(new_x) 
                if is_infnan(f_new_val):
                    if verbose: optimizer.msg(f"[Stopping] Invalid validation objective {f_new_val}", t)
                    break

            x = new_x

            if max_trn_objective is not None and f_new_trn > max_trn_objective:
                f_new_trn = max_trn_objective
                if verbose: optimizer.msg(f"[Clipping] f_new_trn > max_trn_objective, ‖x‖∞={torch.abs(x).max():3.2f}", t)
                break 

            if min_trn_objective is not None and f_new_trn < min_trn_objective:
                f_new_trn = min_trn_objective
                if verbose: optimizer.msg(f"[Clipping] f_new_trn < min_trn_objective, ‖x‖∞={torch.abs(x).max():3.2f}", t)
                break 

            elif abs(f_new_trn - f_cur_trn) < tol:
                if verbose: optimizer.msg(f"[Converged] Training objective change below tol={tol}", t)
                break 

            if validation_objective is not None: 
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

                elif patience_counter >= patience:
                    if verbose:
                        optimizer.msg(f"[Stopping] Validation objective did not improve for {patience} steps (last improvement {t-best_val_iter} steps ago)", t)
                    break
                
                else:
                    patience_counter += 1
            
                    
        else:   # for loop did not break
            if verbose:
                optimizer.msg(f'max_iter {max_iter} reached!')

        if test_objective is not None:
            return Solution(objective=test_objective.get_objective(best_val_x), x=best_val_x, trn_objective=f_cur_trn)
        elif validation_objective is not None:
            return Solution(objective=f_best_val, x=best_val_x, trn_objective=f_cur_trn)
        else:
            return Solution(objective=f_cur_trn, x=x)





if __name__ == "__main__":
    from ep_estimators import Dataset
    if False:
        # Test the optimizers
        for k in OPTIMIZERS.keys():
            break
            print(f"Optimizer: {k}")

            for max_trn_objective in [None, 1e2]:
                for max_val_objective in [None, 1e2]:
                    for objective in [Dataset(g_samples=np.random.randn(100, 10)),
                                Dataset(g_samples=np.random.randn(0, 10))]:

                        x0   = torch.zeros(objective.nobservables, device=objective.device)
                        for minimize in [True, False]:
                            for verbose in [0, 1, 2]:
                                trn1, val1 = objective.split_train_test()
                                trn2, val2, tst2 = objective.split_train_val_test()

                                for (trn, val, tst) in [(objective, None, None), 
                                                        (trn1, val1, None), 
                                                        (trn2, val2, tst2)]:
                                    optimize(x0=x0, objective=trn, optimizer=k, validation_objective=val, test_objective=tst, verbose=verbose, 
                                            minimize=minimize, max_trn_objective=max_trn_objective, max_val_objective=max_val_objective)
                                    
            break # Only test the first optimizer for now

    if False:
        import spin_model
        N    = 100
        J    = spin_model.get_couplings_random(N=N, k=10)
        S, F = spin_model.run_simulation(beta=2, J=J, samples_per_spin=100000)
        i    = 0 # spin index
        g_samples = numpy_to_torch(spin_model.get_g_observables(S, F, i))


        objective = Dataset(g_samples=g_samples)
        x0   = torch.zeros(objective.nobservables, device=objective.device)
        trn2, val2, tst2 = objective.split_train_val_test()
        for k in OPTIMIZERS.keys():
            r = optimize(x0=x0, objective=trn2, optimizer=k, validation_objective=val2, test_objective=tst2, minimize=False, verbose=0)
            print(f"{k:30s} {r.objective:5.3f}")

    class TestObjective(object):
        y = numpy_to_torch(np.array([1,2,3,4]))
        def get_gradient(self, x):
            return x - self.y

        def get_hessian(self, x):
           return torch.eye(len(x), device=x.device)

        def get_objective(self, x):
            return (x - self.y)@(x - self.y)/2

        def split_train_val_test(self):
            return self, None, None
        
    class TestObjectiveNegative(TestObjective):
        def get_objective(self, x):
            return -super().get_objective(x)
        def get_gradient(self, x):
            return -super().get_gradient(x)
        def get_hessian(self, x):   
            return -super().get_hessian(x)
        
    objective = TestObjective()
    x0   = objective.y * 0
    for k in OPTIMIZERS.keys():
        r = optimize(x0=x0, objective=objective, minimize=True, optimizer=k)
        print(f"{k:30s} {r.objective:5.3f} {r.x}")

    objective2 = TestObjectiveNegative()
    x0   = objective2.y * 0
    for k in OPTIMIZERS.keys():
        r = optimize(x0=x0, objective=objective2, minimize=False, optimizer=k)
        print(f"{k:30s} {r.objective:5.3f} {r.x}")
