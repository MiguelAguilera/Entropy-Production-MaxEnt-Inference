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


# ================================================================================
# Dataset class
# This class is used to store the data set of observables samples. It is also used
# to compute the objective function and tilted statistics for the EP estimation methods
# ================================================================================

import functools

class DatasetBase(object):
    @functools.cached_property
    def g_mean(self)     : raise NotImplementedError("g_mean is not implemented")

    @functools.cached_property
    def rev_g_mean(self): raise NotImplementedError("g_rev_mean is not implemented")
    def g_cov(self)     : raise NotImplementedError("g_cov is not implemented")
    def rev_g_cov(self) : raise NotImplementedError("g_rev_cov is not implemented")
    def get_tilted_statistics(self, theta, return_mean=False, return_covariance=False, return_objective=False, **kwargs):
        raise NotImplementedError("get_tilted_statistics is not implemented")

    def get_objective(self, theta):
        # Return objective value for parameters theta
        v = self.get_tilted_statistics(numpy_to_torch(theta), return_objective=True)
        return v['objective']


    @staticmethod
    def _get_trn_indices(nsamples, holdout_fraction, holdout_shuffle=True):
        # Get indices for training and testing data
        assert 0 <= holdout_fraction <= 1, "holdout_fraction must be between 0 and 1"
        trn_nsamples = nsamples - int(nsamples * holdout_fraction)

        if holdout_shuffle:
            perm = np.random.permutation(nsamples)
            trn_indices = perm[:trn_nsamples]
            tst_indices = perm[trn_nsamples:]
        else:
            trn_indices = np.arange(trn_nsamples)
            tst_indices = np.arange(trn_nsamples, nsamples)
        return trn_indices, tst_indices

    def split_train_test(self, holdout_fraction, holdout_shuffle=True):
        raise NotImplementedError("split_train_test is not implemented")



class Dataset(DatasetBase):
    def __init__(self, g_samples, rev_g_samples=None):
        # Arguments:
        #   g_samples                : 2d tensor (nsamples x nobservables) containing samples of observables
        #                              under reverse process 
        #   rev_g_samples            : 2d tensor (nsamples x nobservables) containing samples of observables
        #                              under reverse process . If None, we assume antisymmetric observables
        #                              where rev_g_samples = -g_samples
        self.g_samples        = numpy_to_torch(g_samples)
        self.forward_nsamples = self.g_samples.shape[0]
        self.nobservables     = self.g_samples.shape[1]

        if rev_g_samples is not None:
            self.rev_g_samples = numpy_to_torch(rev_g_samples)
            self.nsamples = self.rev_g_samples.shape[0]
            assert(self.nobservables == self.g_samples.shape[1])
        else:
            self.rev_g_samples = None
            self.nsamples = self.forward_nsamples

        self.device = self.g_samples.device


    @functools.cached_property
    def g_mean(self):
        return self.g_samples.mean(axis=0)


    @functools.cached_property
    def rev_g_mean(self):
        if self.rev_g_samples is None: # antisymmetric observables
            return -self.g_mean
        else:
            return self.get_rev_g_samples().mean(axis=0)


    def get_rev_g_samples(self):
        # Return reverse samples. If not provided, we assume antisymmetric observables
        if self.rev_g_samples is not None:
            return self.rev_g_samples
        else:
            return -self.g_samples


    def g_cov(self):
        # Covariance of g_samples
        return torch.cov(self.g_samples.T, correction=0)
    
    
    def rev_g_cov(self):
        # Covariance of reverse samples
        if self.rev_g_samples is None: # antisymmetric observables, they have the same covariance matrix
            return self.g_cov()
        else:
            torch.cov(self.get_rev_g_samples().T, correction=0)    
    

    def split_train_test(self, holdout_fraction, holdout_shuffle=True):
        # Split current data set into training and heldout testing part
        trn_indices, tst_indices = self._get_trn_indices(self.forward_nsamples, holdout_fraction, holdout_shuffle)

        if self.rev_g_samples is not None:
            trn_indices_rev, tst_indices_rev = self._get_trn_indices(self.nsamples, holdout_fraction, holdout_shuffle)
            rev_trn = self.rev_g_samples[trn_indices_rev]
            rev_tst = self.rev_g_samples[tst_indices_rev]
        else:
            rev_trn, rev_tst = None, None 

        trn = type(self)(g_samples=self.g_samples[trn_indices], rev_g_samples=rev_trn)
        tst = type(self)(g_samples=self.g_samples[tst_indices], rev_g_samples=rev_tst)
        return trn, tst
    

    def get_tilted_statistics(self, theta, return_mean=False, return_covariance=False, return_objective=False, num_chunks=None):
        # Compute tilted statistics under the reverse distribution titled by theta. This may include
        # the objective ( θᵀ<g> - ln(<exp(θᵀg)>_{~p}) ), the tilted mean, and the tilted covariance

        assert return_mean or return_covariance or return_objective

        vals  = {}
        theta = numpy_to_torch(theta)

        with torch.no_grad():
        
            th_g = self.get_rev_g_samples() @ theta

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
                mean = exp_tilt @ self.get_rev_g_samples()
                mean = mean / self.nsamples / norm_const
                vals['tilted_mean'] = mean

                if return_covariance:
                    if num_chunks is None:
                        K = torch.einsum('k,ki,kj->ij', exp_tilt, self.get_rev_g_samples() , self.get_rev_g_samples() )
                        vals['tilted_covariance'] = K / self.nsamples / norm_const - torch.outer(mean, mean)

                    elif num_chunks == -1:
                        vals['tilted_covariance'] = torch.cov(self.get_rev_g_samples().T, correction=0, aweights=exp_tilt)

                    else:
                        # Chunked computation
                        K = torch.zeros((self.nobservables, self.nobservables), dtype=theta.dtype, device=self.device)
                        chunk_size = (self.nsamples + num_chunks - 1) // num_chunks  # Ceiling division

                        for r in range(num_chunks):
                            start = r * chunk_size
                            end = min((r + 1) * chunk_size, self.nsamples)
                            g_chunk = self.get_rev_g_samples()[start:end]
                            
                            K += torch.einsum('k,ki,kj->ij', exp_tilt[start:end], g_chunk, g_chunk)

                        vals['tilted_covariance'] = K / self.nsamples / norm_const - torch.outer(mean, mean)

                        K = torch.einsum('k,ki,kj->ij', exp_tilt, self.get_rev_g_samples() , self.get_rev_g_samples() )

        return vals


    


# TODO : Explain what this does, i.e., calculate statistics directly from state samples
# from forward process. It assumes on antisymmetric observables
class RawDataset(DatasetBase):
    def __init__(self, X0, X1):
        # Arguments:
        # X0          : 2d tensor (nsamples x N) containing initial states of the system
        # X1          : 2d tensor (nsamples x N) containing final states of the system
        
        assert(X0.shape == X1.shape)
        
        self.X0 = numpy_to_torch(X0)
        self.X1 = numpy_to_torch(X1)
        
        # self.N indicates dimensionality of the system
        self.nsamples, self.N = self.X0.shape
        self.nsamples = self.nsamples

        self.device = self.X0.device
    
        self.nobservables = self.g_mean.shape[0]


    @functools.cached_property
    def g_mean(self): # Calculate mean of observables under forward process
        triu_indices = torch.triu_indices(self.N, self.N, offset=1, device=self.device)
        g_mean_raw = (self.X1.T @ self.X0 - self.X0.T @ self.X1 )/ self.nsamples  # shape (N, N)
        return g_mean_raw[triu_indices[0], triu_indices[1]]


    @functools.cached_property
    def rev_g_mean(self): # Calculate mean of observables under reverse process
        return -self.g_mean  # Antisymmetric observables


    def split_train_test(self, holdout_fraction, holdout_shuffle=True):
        # Split current data set into training and heldout testing part
        trn_indices, tst_indices = self._get_trn_indices(self.nsamples, holdout_fraction, holdout_shuffle)
        trn = type(self)(X0=self.X0[trn_indices], X1=self.X1[trn_indices])
        tst = type(self)(X0=self.X0[tst_indices], X1=self.X1[tst_indices])
        return trn, tst


    def get_tilted_statistics(self, theta, return_mean=False, return_covariance=False, return_objective=False, num_chunks=None):
        # Compute tilted statistics under the reverse distribution titled by theta. This may include
        # the objective ( θᵀ<g> - ln(<exp(θᵀg)>_{~p}) ), the tilted mean, and the tilted covariance

        # Here we consider bilinear g_{ij} = x_i * x'_j,
        # using upper-triangular theta without materializing full g_samples.

        if return_covariance:
            raise NotImplementedError("Covariance not implemented for RawDataset")
        
        assert return_objective or return_mean
        
        vals = {}
        theta = numpy_to_torch(theta)

        triu = torch.triu_indices(self.N, self.N, offset=1, device=self.device)

        # 1. θᵀg_k
        Theta = torch.zeros((self.N, self.N), device=self.device)
        Theta[triu[0], triu[1]] = theta
        Y = (Theta - Theta.T) @ self.X1.T
        th_g = torch.sum(self.X0 * Y.T, dim=1)

        th_g_max = torch.max(th_g)
        exp_tilt = torch.exp(th_g - th_g_max)
        norm_const  = torch.mean(exp_tilt)
        weights = exp_tilt / norm_const
        
        if return_objective:
            # To return 'true' normalization constant, we need to correct again for th_g_max
            log_Z = torch.log(norm_const) + th_g_max
            vals['objective'] = float(theta @ self.g_mean - log_Z)

        if return_mean:
            weighted_X = self.X0 * weights[:, None]
            mean_mat = (weighted_X.T @ self.X1) / self.nsamples  
            mean_mat_asymm = mean_mat - mean_mat.T
            g_mean_vec = mean_mat_asymm[triu[0], triu[1]]
            vals['tilted_mean'] = g_mean_vec

        return vals

    
        
class RawDataset2(RawDataset):
    @functools.cached_property
    def g_mean(self): # Calculate mean of observables under forward process
        # Here we consider g_{ij} = (x_i' - x_i) x_j
        return ((self.X1 - self.X0).T @ self.X0 / self.nsamples).flatten()
    

    def get_tilted_statistics(self, theta, return_mean=False, return_covariance=False, return_objective=False, num_chunks=None):
        # Compute tilted statistics under the reverse distribution titled by theta. This may include
        # the objective ( θᵀ<g> - ln(<exp(θᵀg)>_{~p}) ), the tilted mean, and the tilted covariance

        if return_covariance:
            raise NotImplementedError("Covariance not implemented for RawDataset2")
        
        assert return_objective or return_mean
        
        vals = {}
        theta = numpy_to_torch(theta)

        # 1. θᵀg_k 
        theta2d = torch.reshape(theta, (self.N, self.N))
#        theta2d[range(self.N),range(self.N)]=0
        th_g    = torch.einsum('ij,ki,kj->k', theta2d, self.X0 - self.X1, self.X0)

        th_g_max = torch.max(th_g)
        exp_tilt = torch.exp(th_g - th_g_max)
        norm_const = torch.mean(exp_tilt)
        weights = exp_tilt / norm_const
        
        if return_objective:
            # To return 'true' normalization constant, we need to correct again for th_g_max
            log_Z = torch.log(norm_const) + th_g_max
            vals['objective'] = float(theta @ self.g_mean - log_Z)

        if return_mean:
            g_mean = torch.einsum('k,ki,kj->ij', weights, self.X0-self.X1, self.X0)/self.nsamples
            vals['tilted_mean'] = g_mean.flatten()

        return vals




# EPEstimators return Solution namedtuples such as the following
#   objective (float) : estimate of EP
#   theta (torch tensor of length nobservables) : optimal conjugate parameters
#   tst_objective (float) : estimate of EP on heldout test data (if holdout is used)
Solution = namedtuple('Solution', ['objective', 'theta', 'tst_objective'], defaults=[None])

# ==========================================
# Entropy production (EP) estimation methods 
# ==========================================
class EPEstimators(object):
    # EP estimators based on optimizing our variational principle
    #  
    def __init__(self, data, use_upper_only=False):
        """
        Parameters:
            data              : data object
            use_upper_only    : if True, interpret theta as flattened upper-triangle (i<j) of n x n
        """
        self.data = data


    def get_valid_solution(self, objective, theta, tst_objective=None):
        # This returns a Solution object, after doing some basic sanity checking of the values
        # This checking is useful in the undersampled regime with many dimensions and few samles
        if objective < 0:
            # EP estimate should never be negative, as we can always achieve objective=0 with all 0s theta
            objective = 0.0 
            if theta is not None:
                theta = 0*theta
        elif objective >= np.log(self.data.nsamples):
            # EP estimate should not be larger than log(# samples), because it is not possible
            # to estimate KL divergence larger than log(m) from m samples
            objective = np.log(self.data.nsamples)
        return Solution(objective=objective, theta=theta, tst_objective=tst_objective)


    def get_EP_Newton(self, 
                      holdout=False, holdout_fraction=1/2, holdout_shuffle=True, verbose=0,
                      max_iter=1000, tol=1e-4, linsolve_eps=1e-4, num_chunks=None,
                      trust_radius=None, solve_constrained=True, adjust_radius=False,
                      eta0=0.0, eta1=0.25, eta2=0.75, trust_radius_min=1e-3, trust_radius_max=1000.0, trust_radius_adjust_max_iter=100):
        # Estimate EP by optimizing objective using Newton's method. We support advanced
        # features like Newton's method with trust region constraints
        #
        # Arguments (all optional):
        #   holdout (bool)   : if True, we split the data into training and testing sets
        #   holdout_fraction : fraction of samples for test split
        #   holdout_shuffle  : shuffle data before splitting
        #   verbose (int)    : Verbosity level for printing debugging information (0=no printing)
        #   max_iter (int)   : maximum number of iterations
        #   tol (float)      : early stopping once objective does not improve by more than tol
        #   linsolve_eps     : regularization parameter for linear solvers (helps numerical stability)
        #   num_chunks       : chunk size for memory-efficient covariance computation
        #
        # Trust-region-related arguments (all optional):
        #   trust_radius (float or None) : maximum trust region (if None, trust region constraint are not used)
        #   solve_constrained (bool)     : if True, we find direction by solving the constrained trust-region problem 
        #                                  if False, we simply rescale usual Newton step to desired maximum norm 
        #   adjust_radius (bool)         : change trust-region radius in an adaptive way
        #   eta0=0.0, eta1=0.25, eta2=0.75,trust_radius_min,trust_radius_max,trust_radius_adjust_max_iter
        #                                : hyperparameters for adjusting trust radius

        funcname = 'get_EP_Newton_steps'

        with torch.no_grad():   # We don't need to torch to keep track of gradients (sometimes a bit faster)

            if holdout:
                trn, tst = self.data.split_train_test(holdout_fraction, holdout_shuffle)
                f_cur_trn = f_cur_tst = f_new_trn = f_new_tst = 0.0
            else:
                trn = self.data
                f_cur_trn = f_new_trn = 0.0
            
            theta = torch.zeros(trn.nobservables, device=trn.device)
            I     = torch.eye(len(theta), device=theta.device)

            for t in range(max_iter):
                if verbose and verbose > 1: 
                    print(f'{funcname} : iteration {t:5d} f_cur_trn={f_cur_trn: 3f}', f'f_cur_tst={f_cur_tst: 3f}' if holdout else '')

                # Find Newton step direction. We first get gradient and Hessian
                stats = trn.get_tilted_statistics(theta, return_mean=True, return_covariance=True, num_chunks=num_chunks)
                g_theta = stats['tilted_mean']
                H_theta = stats['tilted_covariance']

                if is_infnan(H_theta.sum()): 
                    # Error occured, usually it means theta is too big
                    if verbose: print('{funcname} : [Stopping] Invalid Hessian')
                    break

                grad = trn.g_mean - g_theta
                H_theta += linsolve_eps * I  # regularize Hessian by adding a small I

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
                        if verbose: print("{funcname} : max_iter reached in adjust_radius loop!")

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

                if is_infnan(f_new_trn):  
                    if verbose: print(f"{funcname} : [Stopping] Invalid value {f_new_trn} in training objective at iter {t}")
                    break                  # Training value should be finite and increasing
                elif f_new_trn <= f_cur_trn:
                    if verbose: print(f"[Stopping] Training objective did not improve (f_new_trn <= f_cur_trn) at iter {t}")
                    break
                elif np.abs(f_new_trn - f_cur_trn) <= tol: 
                    if verbose: print(f"{funcname} : [Converged] Train objective change below tol={tol} at iter {t}")
                    last_round = True      # Break when training objective stops improving by more than tol
                elif f_new_trn > np.log(trn.nsamples):  
                    # One cannot reliably estimate KL divergence larger than log(# samples).
                    # This is a signature of undersampling; when it happens, we clip our estimate of the 
                    # objective and exit
                    if verbose: print(f"{funcname} : [Clipping] Training objective exceeded log(#samples), clipping to log(nsamples) at iter {t}")
                    f_new_trn = np.log(trn.nsamples)
                    last_round = True

                if holdout:                # Do the same checks but now on the heldout test data
                    f_new_tst = tst.get_objective(new_theta) 
                    if is_infnan(f_new_tst):
                        if verbose: print(f"{funcname} : [Stopping] Invalid value {f_new_tst} in test objective at iter {t}")
                        break
                    elif f_new_tst <= f_cur_tst:
                        if verbose: print(f"[Stopping] Testing objective did not improve (f_new_tst <= f_cur_tst) at iter {t}")
                        break
                    elif np.abs(f_new_tst - f_cur_tst) <= tol:
                        if verbose: print(f"{funcname} : [Converged] Test objective change below tol={tol} at iter {t}")
                        last_round = True
                    elif f_new_tst > np.log(tst.nsamples):
                        if verbose: print(f"{funcname} : [Clipping] Test objective exceeded log(#samples), clipping at iter {t}")
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
                objective = self.data.get_objective(theta)
                return self.get_valid_solution(objective=objective, theta=theta, tst_objective=f_cur_tst)
            else:
                return self.get_valid_solution(objective=f_cur_trn, theta=theta)



    def get_EP_GradientAscent(self, theta_init=None, 
                              holdout=False, holdout_fraction=1/2, holdout_shuffle=True, 
                              lr=0.01, max_iter=1000, patience = 10, tol=1e-4, verbose=0,
                              use_Adam=True, beta1=0.9, beta2=0.999, skip_warm_up=False):
        # Estimate EP using gradient ascent algorithm

        # Arguments:
        #   theta_init       : torch tensor (of length (nspins-1)), specifiying initial parameters (set to 0s if None)
        #   holdout (bool)   : if True, we split the data into training and testing sets
        #   holdout_fraction : fraction of samples for test split
        #   holdout_shuffle  : shuffle data before splitting
        #   lr               : learning rate
        #   max_iter (int)   : maximum number of iterations
        #   tol (float)      : early stopping once objective does not improve by more than tol
        #   verbose (int)    : Verbosity level for printing debugging information (0=no printing)
        #
        # Adam-related arguments:
        #   use_Adam       : whether to use use_Adam algorithm w/ momentum or just regular grad. ascent
        #   beta1, beta2   : Adam moment decay parameters
        #   skip_warm_up   : Adam option
        #
        # Returns:
        #   Solution object with objective (EP estimate), theta, and tst_objective (if holdout)

        funcname = 'get_EP_GradientAscent'
        report_every = 10  # if verbose > 1, we report every report_every iterations

        with torch.no_grad():  # We calculate our own gradients, so we don't need to torch to do it (sometimes a bit faster)

            if holdout:
                trn, tst = self.data.split_train_test(holdout_fraction, holdout_shuffle)
                f_cur_trn, f_cur_tst = np.nan, np.nan
            else:
                trn = self.data
                f_cur_trn = np.nan
            
            if theta_init is not None:
                new_theta = theta_init
            else:
                new_theta = torch.zeros(trn.nobservables, device=trn.device)

            m = torch.zeros_like(new_theta)
            v = torch.zeros_like(new_theta)
            
            best_tst_score   = -float('inf')  # for maximization
            patience_counter = 0
            old_time         = time.time()
            for t in range(max_iter):
                if verbose and verbose > 1 and t % report_every == 0:
                    new_time = time.time()
                    print(f'{funcname} : iteration {t:5d} | {(new_time - old_time)/report_every:3f}s/iter | f_cur_trn={f_cur_trn: 3f}', f'f_cur_tst={f_cur_tst: 3f}' if holdout else '')
                    old_time = new_time

                tilted_stats = trn.get_tilted_statistics(new_theta, return_objective=True, return_mean=True)
                f_new_trn    = tilted_stats['objective']

                grad         = trn.g_mean - tilted_stats['tilted_mean']   # Gradient

                # Different conditions that will stop optimization. See get_EP_Newton above
                # for a description of different branches
                last_round = False # flag that indicates whether to break after updating values
                if is_infnan(f_new_trn):
                    if verbose: print(f"{funcname} : [Stopping] Invalid value {f_new_trn} in training objective at iter {t}")
                    break
#                elif f_new_trn <= f_cur_trn and t > min_iter:
#                    print(f"[Stopping] Training objective did not improve (f_new_trn <= f_cur_trn) at iter {t}")
#                    break
                elif f_new_trn > np.log(trn.nsamples):
                    if verbose: print(f"{funcname} : [Clipping] Training objective exceeded log(#samples), clipping to log(nsamples) at iter {t}")
                    f_new_trn = np.log(trn.nsamples)
                    last_round = True
                elif np.abs(f_new_trn - f_cur_trn) <= tol:
                    if verbose: print(f"{funcname} : [Converged] Training objective change below tol={tol} at iter {t}")
                    last_round = True

                if holdout:
                    f_new_tst = tst.get_objective(new_theta) 
                    train_gain = f_new_trn - f_cur_trn
                    test_drop = f_cur_tst - f_new_tst
                    if is_infnan(f_new_tst):
                        if verbose: print(f"{funcname} : [Stopping] Invalid value {f_new_tst} in test objective at iter {t}")
                        break
#                    elif test_drop > 0 and train_gain > tol :
#                        print(f"[Stopping] Test objective did not improve (f_new_tst <= f_cur_tst and (f_new_trn - f_cur_trn) > tol ) at iter {t}")
#                        break
                    elif f_new_tst > np.log(tst.nsamples):
                        if verbose: print(f"{funcname} : [Clipping] Test objective exceeded log(#samples), clipping at iter {t}")
                        f_new_tst = np.log(tst.nsamples)
                        last_round = True
                    elif np.abs(f_new_tst - f_cur_tst) <= tol:
                        if verbose: print(f"{funcname} : [Converged] Test objective change below tol={tol} at iter {t}")
                        last_round = True
                    
                    if f_new_tst > best_tst_score:
                        best_tst_score = f_new_tst
                        best_theta = new_theta.clone()  # Save the best model
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= patience:
                        if verbose: print(f"{funcname} : [Stopping] Test objective did not improve  (f_new_tst <= f_cur_tst and)  for {patience} steps iter {t}")
                        theta     = best_theta.clone()
                        f_cur_tst = best_tst_score
                        break
                        
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
                objective = self.data.get_objective(theta)
                return self.get_valid_solution(objective=objective, theta=theta, tst_objective=f_cur_tst)
            else:
                return self.get_valid_solution(objective=f_cur_trn, theta=theta)



    # Estimate EP using the multidimensional TUR method
    def get_EP_MTUR(self, num_chunks=None, linsolve_eps=1e-4):
        # The MTUR is defined as (1/2) (<g>_(p - ~p))^T K^-1 (<g>_p - <g>_(p - ~p))
        # where μ = (p + ~p)/2 is the mixture of the forward and reverse distributions
        # and K^-1 is covariance matrix of g under (p + ~p)/2.
        # 
        # Optional arguments
        #   num_chunks (int)         : chunk covariance computations to reduce memory requirements
        #   linsolve_eps (float)     : regularization parameter for covariance matrices, used to improve
        #                               numerical stability of linear solvers

        obj = self.data
        mean_diff    = obj.g_mean - obj.rev_g_mean

        combined_mean    = (obj.g_mean + obj.rev_g_mean)/2

        # now compute covariance of (p + ~p)/2
        if num_chunks == -1:
            second_moment_forward = obj.g_cov() + torch.outer(obj.g_mean, obj.g_mean)
            second_moment_reverse = obj.rev_g_cov() + torch.outer(obj.rev_g_mean, obj.rev_g_mean)
            combined_cov = (second_moment_forward + second_moment_reverse)/2 - torch.outer(combined_mean, combined_mean)
        
        else:
            combined_samples = torch.concatenate([obj.g_samples, obj.get_rev_g_samples()], dim=0)
            num_total        = obj.nsamples + obj.nsamples
            weights          = torch.empty(num_total)
            weights[:obj.nsamples] = 1.0/obj.nsamples/2.0
            weights[obj.nsamples:] = 1.0/obj.nsamples/2.0

            if num_chunks is None:
                combined_cov = torch.einsum('k,ki,kj->ij', weights, combined_samples, combined_samples)

            else:
                # Chunked computation
                combined_cov = torch.zeros((obj.nobservables, obj.nobservables), device=obj.device)
                chunk_size = (num_total + num_chunks - 1) // num_chunks  # Ceiling division

                for r in range(num_chunks):
                    start = r * chunk_size
                    end = min((r + 1) * chunk_size, num_total)
                    chunk = combined_samples[start:end]
                    
                    combined_cov += torch.einsum('k,ki,kj->ij', weights[start:end], chunk, chunk)

        x  = solve_linear_psd(combined_cov + linsolve_eps*eye_like(combined_cov), mean_diff)
        objective = float(x @ mean_diff)/2

        return self.get_valid_solution(objective=objective, theta=None)
