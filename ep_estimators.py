# ============================================================
# Contains methods to estimate EP in multipartite spin systems
# ============================================================

# The key datastructure here is S. This is a 2d torch tensor of -1 and 1s
# whose shape is (number of samples, number of spins). Each row indicates
# the state of the system right before spin i changed state. 

# The observables are g_{ij} = (x_i' - x_i) x_j where x_i' and x_i 
# indicates the state of spin i after and before the jump, 
# x_j is the state of every other spin


import os, time
import numpy as np
from collections import namedtuple

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]='1'
import torch

from utils import *


def get_EP_MTUR(g_samples, rev_g_samples, num_chunks=None, linsolve_eps=1e-4):
    # Estimate EP using the multidimensional TUR method
    #
    # The MTUR is defined as (1/2) (<g>_(p - ~p))^T K^-1 (<g>_p - <g>_(p - ~p))
    # where μ = (p + ~p)/2 is the mixture of the forward and reverse distributions
    # and K^-1 is covariance matrix of g under (p + ~p)/2.
    # 
    # Arguments
    #   g_samples                : 2d tensor (nsamples x nobservables) containing samples of observables
    #                              under reverse process 
    #   rev_g_samples            : 2d tensor (nsamples x nobservables) containing samples of observables
    #                              under reverse process 
    # Optional arguments
    #   num_chunks (int)         : chunk covariance computations to reduce memory requirements
    #   linsolve_eps (float)     : regularization parameter for covariance matrices, used to improve
    #                               numerical stability of linear solvers

    g_samples     = numpy_to_torch(g_samples)
    rev_g_samples = numpy_to_torch(rev_g_samples)

    g_mean       = g_samples.mean(axis=0) 
    rev_g_mean   = rev_g_samples.mean(axis=0)
    mean_diff    = g_mean - rev_g_mean

    # now compute covariance of (p + ~p)/2
    combined_samples = torch.concatenate([g_samples, rev_g_samples], dim=0)
    combined_mean    = (g_mean + rev_g_mean)/2

    num_forward      = g_samples.shape[0] 
    num_reverse      = rev_g_samples.shape[0]
    num_total        = num_forward + num_reverse
    weights          = torch.empty(num_forward + num_reverse)
    weights[:num_forward] = 1.0/num_forward/2.0
    weights[num_forward:] = 1.0/num_reverse/2.0

    if num_chunks is None:
        combined_cov = torch.einsum('k,ki,kj->ij', weights, combined_samples, combined_samples)

    else:
        # Chunked computation
        num_observables = g_samples.shape[1]
        combined_cov = torch.zeros((num_observables, num_observables), device=g_samples.device)
        chunk_size = (num_total + num_chunks - 1) // num_chunks  # Ceiling division

        for r in range(num_chunks):
            start = r * chunk_size
            end = min((r + 1) * chunk_size, num_total)
            chunk = combined_samples[start:end]
            
            combined_cov += torch.einsum('k,ki,kj->ij', weights[start:end], chunk, chunk)

    combined_cov += linsolve_eps*eye_like(combined_cov)
    x  = solve_linear_psd(combined_cov, mean_diff)
    return float(x @ mean_diff)/2



# EPEstimators return Solution namedtuples such as the following
#   objective (float) : estimate of EP
#   theta (torch tensor of length nobservables) : optimal conjugate parameters
#   tst_objective (float) : estimate of EP on heldout test data (if holdout is used)
Solution = namedtuple('Solution', ['objective', 'theta', 'tst_objective'], defaults=[None])


def theta_to_upper_matrix(theta, n):
    """
    Map a vectorized upper triangle (i < j) into a full n x n matrix.
    The diagonal and lower triangle are set to 0.

    Parameters:
        theta : (n*(n-1)//2,) vector
        n     : number of spins

    Returns:
        Theta : (n, n) matrix with theta values in the upper triangle
    """
    Theta = torch.zeros((n, n), device=theta.device)
    triu_indices = torch.triu_indices(n, n, offset=1)
    Theta[triu_indices[0], triu_indices[1]] = theta
    return Theta

def tilted_statistics_bilinear_upper(X, Xp, theta, return_mean=False, return_covariance=False, return_objective=False, g_mean=None):
    """
    Computes tilted statistics (objective, mean, covariance) for bilinear g_{ij} = x_i * x'_j,
    using upper-triangular theta without materializing full g_samples.
    """
    n = X.shape[1]
    nsamples = X.shape[0]
    triu = torch.triu_indices(n, n, offset=1)
    
    # 1. θᵀg_k
    Theta = torch.zeros((n, n), device=theta.device)
    Theta[triu[0], triu[1]] = theta
    Y = (Theta - Theta.T) @ Xp.T
    th_g = torch.sum(X * Y.T, dim=1)

    th_g_max = torch.max(th_g)
    exp_tilt = torch.exp(th_g - th_g_max)
    norm_const  = torch.mean(exp_tilt)
    weights = exp_tilt / norm_const
    vals = {}
    
    if return_objective:
        # To return 'true' normalization constant, we need to correct again for th_g_max
        log_Z = torch.log(norm_const) + th_g_max
        vals['objective'] = float(theta @ g_mean - log_Z)

    if return_mean or return_covariance:
        weighted_X = X * weights[:, None]
        mean_mat = (weighted_X.T @ Xp) / nsamples / norm_const  # shape (n, n)
        mean_mat_asymm = mean_mat - mean_mat.T
        g_mean_vec = mean_mat_asymm[triu[0], triu[1]]
        vals['tilted_mean'] = g_mean_vec

    return vals

    
        
class EPEstimators(object):
    # EP estimators based on optimizing our variational principle
    #  
    def __init__(self, g_mean, rev_g_samples=None, tilted_statistics_function=None,
                 X=None, Xp=None, use_upper_only=False,
                 holdout_fraction=0.5, holdout_shuffle=False,
                 num_chunks=None, linsolve_eps=1e-4):
        """
        Parameters:
            g_mean            : 1d tensor of length nobservables
            rev_g_samples     : optional 2d tensor (nsamples x nobservables)
            rev_g_function    : optional function(X, Xp) returning rev_g_samples or sufficient statistics
            X, Xp             : required if rev_g_function is used
            use_upper_only    : if True, interpret theta as flattened upper-triangle (i<j) of n x n
            holdout_fraction  : fraction of samples for test split
            holdout_shuffle   : shuffle data before splitting
            num_chunks        : chunk size for memory-efficient covariance computation
            linsolve_eps      : regularization for linear solves
        """

        self.g_mean = numpy_to_torch(g_mean)
        self.linsolve_eps = linsolve_eps
        self.num_chunks = num_chunks
        self.holdout_fraction = holdout_fraction
        self.holdout_shuffle = holdout_shuffle
        self.use_upper_only = use_upper_only

        if rev_g_samples is not None:
            self.rev_g_samples = numpy_to_torch(rev_g_samples)
            self.nsamples, self.nobservables = self.rev_g_samples.shape
            self.device = self.rev_g_samples.device
            self.use_function = False

        elif tilted_statistics_function is not None:
            assert X is not None and Xp is not None, "Must provide X and Xp when using rev_g_function"
            self.tilted_statistics_function = tilted_statistics_function
            self.X = numpy_to_torch(X)
            self.Xp = numpy_to_torch(Xp)
            self.nsamples, self.nspins = self.X.shape
            self.nobservables = self.g_mean.shape[0]
            self.device = self.X.device
            self.use_function = True

        else:
            raise ValueError("Must provide either rev_g_samples or rev_g_function")

        assert 0 <= holdout_fraction <= 1, "holdout_fraction must be between 0 and 1"
        assert linsolve_eps >= 0, "linsolve_eps must be non-negative"

        self._init_args = [
            'g_mean', 'holdout_fraction', 'holdout_shuffle',
            'num_chunks', 'linsolve_eps', 'use_upper_only'
        ]

        
    def split_train_test(self):
        # Split current data set into training and heldout testing part
        if not hasattr(self, 'trn_tst_split_'):

            trn_nsamples = self.nsamples - int(self.nsamples * self.holdout_fraction)

            kw_args = {k: getattr(self, k) for k in self._init_args}

            if not self.use_function:
                if self.holdout_shuffle:
                    perm = np.random.permutation(self.nsamples)
                    rev_g_samples = self.rev_g_samples[perm]
                else:
                    rev_g_samples = self.rev_g_samples

                trn = EPEstimators(
                    rev_g_samples=rev_g_samples[:trn_nsamples],
                    **kw_args
                )
                tst = EPEstimators(
                    rev_g_samples=rev_g_samples[trn_nsamples:],
                    **kw_args
                )

            else:
                if self.holdout_shuffle:
                    perm = np.random.permutation(self.nsamples)
                    X = self.X[perm]
                    Xp = self.Xp[perm]
                else:
                    X = self.X
                    Xp = self.Xp

                trn = EPEstimators(
                    X=X[:trn_nsamples], Xp=Xp[:trn_nsamples],
                    tilted_statistics_function=self.tilted_statistics_function,
                    **kw_args
                )
                tst = EPEstimators(
                    X=X[trn_nsamples:], Xp=Xp[trn_nsamples:],
                    tilted_statistics_function=self.tilted_statistics_function,
                    **kw_args
                )

            self.trn_tst_split_ = trn, tst

        return self.trn_tst_split_


    
    # ====================================================================================
    # Methods to compute observable statistics under tilted reverse distribution
    # ====================================================================================
    def _get_tilted_statistics(self, theta, return_mean=False, return_covariance=False, return_objective=False):
        # This internal method computes tilted statistics reverse distribution titled by theta. This may include
        # the objective ( θᵀ<g> - ln(<exp(θᵀg)>_{~p}) ), the tilted mean, and the tilted covariance

        assert return_mean or return_covariance or return_objective


        if hasattr(self, 'tilted_statistics_function') and self.tilted_statistics_function is not None:
            return self.tilted_statistics_function(
                X=self.X, Xp=self.Xp, theta=theta,
                return_mean=return_mean,
                return_covariance=return_covariance,
                return_objective=return_objective,
                g_mean=self.g_mean
            )

        vals       = {}

        with torch.no_grad():
        
            th_g = self.rev_g_samples @ theta

            # To improve numerical stability, the exponentially tilting discounts exp(-th_g_max)
            # The same multiplicative corrections enters into the normalization constant and the tilted
            # means and covariance, so its cancels out
            th_g_max    = torch.max(th_g)
            exp_tilt    = torch.exp(th_g - th_g_max)
            norm_const  = torch.mean(exp_tilt)

            if return_objective:
                # To return 'true' normalization constant, we need to correct again for th_g_max
                log_Z             = torch.log(norm_const) + th_g_max
                vals['objective'] = float( theta @ self.g_mean - log_Z )

            if return_mean or return_covariance:
                mean = exp_tilt @ self.rev_g_samples / self.nsamples / norm_const
                vals['tilted_mean'] = mean

                if return_covariance:
                    if self.num_chunks is None:
                        K = torch.einsum('k,ki,kj->ij', exp_tilt, self.rev_g_samples, self.rev_g_samples)

                    else:
                        # Chunked computation
                        K = torch.zeros((self.nobservables, self.nobservables), device=self.device)
                        chunk_size = (self.nsamples + self.num_chunks - 1) // self.num_chunks  # Ceiling division

                        for r in range(self.num_chunks):
                            start = r * chunk_size
                            end = min((r + 1) * chunk_size, self.nsamples)
                            g_chunk = self.rev_g_samples[start:end]
                            
                            K += torch.einsum('k,ki,kj->ij', exp_tilt[start:end], g_chunk, g_chunk)

                    vals['tilted_covariance'] = K / self.nsamples / norm_const - torch.outer(mean, mean)

        return vals

    def get_objective(self, theta):
        # Return objective value for parameters theta
        v = self._get_tilted_statistics(theta, return_objective=True)
        return v['objective']


    # ==========================================
    # Entropy production (EP) estimation methods 
    # ==========================================

    def get_valid_solution(self, objective, theta, tst_objective=None):
        # This returns a Solution object, after doing some basic sanity checking of the values
        # This checking is useful in the undersampled regime with many dimensions and few samles
        if objective < 0:
            # EP estimate should never be negative, as we can always achieve objective=0 with all 0s theta
            objective = 0.0 
            if theta is not None:
                theta = 0*theta
        elif objective >= np.log(self.nsamples):
            # EP estimate should not be larger than log(# samples), because it is not possible
            # to estimate KL divergence larger than log(m) from m samples
            objective = np.log(self.nsamples)
        return Solution(objective=objective, theta=theta, tst_objective=tst_objective)


    def get_EP_Newton(self, max_iter=1000, tol=1e-4, holdout=False, verbose=False,
        trust_radius=None, solve_constrained=True, adjust_radius=False,
        eta0=0.0, eta1=0.25, eta2=0.75, trust_radius_min=1e-3, trust_radius_max=1000.0, trust_radius_adjust_max_iter=100):
        """
        Estimate EP by optimizing objective using Newton's method. We support advanced
        features like Newton's method with trust region constraints

        Arguments:
          max_iter (int) : maximum number of iterations
          tol (float)    : early stopping once objective does not improve by more than tol
          holdout (bool) : whether to use holdout data for early stopping
          verbose (bool) : print debugging information

        Trust-region-related arguments:
          trust_radius (float or None) : maximum trust region (if None, trust region constraint are not used)
          solve_constrained (bool)     : if True, we find direction by solving the constrained trust-region problem 
                                         if False, we simply rescale usual Newton step to desired maximum norm 
          adjust_radius (bool)         : change trust-region radius in an adaptive way
          eta0=0.0, eta1=0.25, eta2=0.75,trust_radius_min,trust_radius_max,trust_radius_adjust_max_iter
                                       : hyperparameters for adjusting trust radius
        """

        with torch.no_grad():   # We don't need to torch to keep track of gradients (sometimes a bit faster)

            if holdout:
                trn, tst = self.split_train_test()
                f_cur_trn = f_cur_tst = f_new_trn = f_new_tst = 0.0
            else:
                trn = self
                f_cur_trn = f_new_trn = 0.0
            
            theta = torch.zeros(self.nobservables, device=self.device)
            I     = torch.eye(self.nobservables, device=self.device)

            for _ in range(max_iter):
                # Find Newton step direction. We first get gradient and Hessian
                stats = trn._get_tilted_statistics(theta, return_mean=True, return_covariance=True)
                g_theta = stats['tilted_mean']
                H_theta = stats['tilted_covariance']

                if is_infnan(H_theta.sum()): 
                    # Error occured, usually it means theta is too big
                    if verbose: print('invalid Hessian in get_EP_Newton_steps')
                    break

                grad = trn.g_mean - g_theta
                H_theta += self.linsolve_eps * I  # regularize Hessian by adding a small I

                if trust_radius is not None and solve_constrained:
                    # Solve the constrained trust region problem using the
                    # Steihaug-Toint Truncated Conjugate-Gradient Method

                    for _ in range(trust_radius_adjust_max_iter): # loop until trust_radius is adjusted properly
                        delta_theta = steihaug_toint_cg(A=H_theta, b=grad, trust_radius=trust_radius)
                        new_theta  = theta + delta_theta
                        f_new_trn  = trn.get_objective(new_theta)

                        if not adjust_radius:    
                            # We don't care about adjust trust_radius, so we just accept the current direction
                            break

                        else:
                            pred_red = grad @ delta_theta + 0.5 * delta_theta @ (H_theta @ delta_theta)

                            act_red = f_new_trn - f_cur_trn
                            rho = act_red / (pred_red + 1e-20)

                            assert not is_infnan(rho), "rho is not a valid number in adjust_radius code. Try disabling adjust_radius=False"
                            if rho > eta0:      # accept new theta
                                break
                            if trust_radius < trust_radius_min:
                                trust_radius = trust_radius_min
                                break

                            if rho < eta1:
                                trust_radius *= 0.25
                            elif rho > eta2 and delta_theta.norm() >= trust_radius:
                                trust_radius = min(2.0 * trust_radius, trust_radius_max)


                    else: # for loop finished without breaking
                        if verbose: 
                            print("max_iter reached in adjust_radius loop!")

                else:
                    # Find regular Newton direction
                    delta_theta = solve_linear_psd(A=H_theta, b=grad)
                    if trust_radius is not None:
                        # We do a quick-and-dirty approximation to trust-region constraint
                        # by rescaling norm of Newton step if it is too large
                        delta_theta *= trust_radius/max(trust_radius, delta_theta.norm())
                    new_theta  = theta + delta_theta
                    f_new_trn  = trn.get_objective(new_theta)


                last_round = False # set to True if we want break after updating theta and objective values

                if is_infnan(f_new_trn) or f_new_trn <= f_cur_trn:  
                    break                  # Training value should be finite and increasing
                elif np.abs(f_new_trn - f_cur_trn) <= tol: 
                    last_round = True      # Break when training objective stops improving by more than tol
                elif f_new_trn > np.log(trn.nsamples):  
                    # One cannot reliably estimate KL divergence larger than log(# samples).
                    # This is a signature of undersampling; when it happens, we clip our estimate of the 
                    # objective and exit
                    f_new_trn = np.log(trn.nsamples)
                    last_round = True

                if holdout:                # Do the same checks but now on the heldout test data
                    f_new_tst = tst.get_objective(new_theta) 
                    if is_infnan(f_new_tst) or f_new_tst <= f_cur_tst:
                        break
                    elif np.abs(f_new_tst - f_cur_tst) <= tol:
                        last_round = True
                    elif f_new_tst > np.log(tst.nsamples):
                        f_new_tst = np.log(tst.nsamples)
                        last_round = True

                # Update our estimate of the paramters and objective value
                f_cur_trn, theta = f_new_trn, new_theta
                if holdout:
                    f_cur_tst = f_new_tst

                if last_round:
                    break

            else:   # for loop did not break
                if max_iter > 10 and verbose:
                    # print warning about max iterations reached, but only if its a large number
                    print(f'max_iter {max_iter} reached in get_EP_Newton!')
                pass

            if holdout:
                return self.get_valid_solution(objective=self.get_objective(theta), theta=theta, tst_objective=f_cur_tst)
            else:
                return self.get_valid_solution(objective=f_cur_trn, theta=theta)



    def get_EP_GradientAscent(self, theta_init=None, holdout=False, lr=0.01, max_iter=1000, min_iter=100, tol=1e-4, verbose=False,
                              use_Adam=True, beta1=0.9, beta2=0.999, skip_warm_up=False):
        """
        Estimate EP using gradient ascent algorithm

        Arguments:
          theta_init     : torch tensor (of length (nspins-1)), specifiying initial parameters (set to 0s if None)
          holdout (bool) : whether to use holdout data for early stopping
          lr             : learning rate
          max_iter (int) : maximum number of iterations
          tol (float)    : early stopping once objective does not improve by more than tol
          verbose (bool) : print debugging information

        Adam-related arguments:
          use_Adam       : whether to use use_Adam algorithm w/ momentum or just regular grad. ascent
          beta1, beta2   : Adam moment decay parameters
          skip_warm_up   : Adam option
        
        Returns:
          Solution object with objective (EP estimate), theta, and tst_objective (if holdout)
        """

        with torch.no_grad():  # We calculate our own gradients, so we don't need to torch to do it (sometimes a bit faster)

            if holdout:
                trn, tst = self.split_train_test()
                f_cur_trn, f_cur_tst = np.nan, np.nan
            else:
                trn = self
                f_cur_trn = np.nan
            
            if theta_init is not None:
                new_theta = theta_init
            else:
                new_theta = torch.zeros(self.nobservables, device=self.device)

            m = torch.zeros_like(new_theta)
            v = torch.zeros_like(new_theta)
            
            for t in range(max_iter):
                tilted_stats = trn._get_tilted_statistics(new_theta, return_objective=True, return_mean=True)
                f_new_trn    = tilted_stats['objective']

                grad         = trn.g_mean - tilted_stats['tilted_mean']   # Gradient


                # Different conditions that will stop optimization. See get_EP_Newton above
                # for a description of different branches
                last_round = False # flag that indicates whether to break after updating values
                if is_infnan(f_new_trn):
                    print(f"[Stopping] Invalid value (NaN or Inf) in training objective at iter {t}")
                    break
#                elif f_new_trn <= f_cur_trn and t > min_iter:
#                    print(f"[Stopping] Training objective did not improve (f_new_trn <= f_cur_trn) at iter {t}")
#                    break
                elif f_new_trn > np.log(trn.nsamples):
                    print(f"[Clipping] Training objective exceeded log(#samples), clipping to log(nsamples) at iter {t}")
                    f_new_trn = np.log(trn.nsamples)
                    last_round = True
                elif np.abs(f_new_trn - f_cur_trn) <= tol:
                    print(f"[Converged] Training objective change below tol={tol} at iter {t}")
                    last_round = True

                if holdout:
                    f_new_tst = tst.get_objective(new_theta) 
                    train_gain = f_new_trn - f_cur_trn
                    test_drop = f_cur_tst - f_new_tst
                    if is_infnan(f_new_tst):
                        print(f"[Stopping] Invalid value (NaN or Inf) in test objective at iter {t}")
                        break
                    elif test_drop > 0 and train_gain > tol :
                        print(f"[Stopping] Test objective did not improve (f_new_tst <= f_cur_tst and (f_new_trn - f_cur_trn) > tol ) at iter {t}")
                        break
                    elif f_new_tst > np.log(tst.nsamples):
                        print(f"[Clipping] Test objective exceeded log(#samples), clipping at iter {t}")
                        f_new_tst = np.log(tst.nsamples)
                        last_round = True
                    elif np.abs(f_new_tst - f_cur_tst) <= tol:
                        print(f"[Converged] Test objective change below tol={tol} at iter {t}")
                        last_round = True

                f_cur_trn, theta = f_new_trn, new_theta
                if holdout:
                    f_cur_tst = f_new_tst

                if last_round:
                    break


                if use_Adam:
                    # Adam moment updates
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad**2)
                    if skip_warm_up:
                        m_hat = m
                        v_hat = v
                    else:
                        m_hat = m / (1 - beta1 ** (t+1))
                        v_hat = v / (1 - beta2 ** (t+1))

                    # Compute parameter update
                    delta_theta = lr * m_hat / (v_hat.sqrt() + 1e-8)

                    new_theta += delta_theta

                else:
                    # regular gradient ascent
                    delta_theta = lr * grad

                new_theta = theta + delta_theta

            else:   # for loop did not break
                if verbose:
                    # print warning about max iterations reached, but only if its a large number
                    print(f'max_iter {max_iter} reached in get_EP_GradientAscent!')
                pass

            if holdout:
                return self.get_valid_solution(objective=self.get_objective(theta), theta=theta, tst_objective=f_cur_tst)
            else:
                return self.get_valid_solution(objective=f_cur_trn, theta=theta)

