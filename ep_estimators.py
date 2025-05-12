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
    def _get_trn_indices(nsamples, holdout_fraction=1/2, holdout_shuffle=True, with_replacement=False):
        # Get indices for training and testing data
        assert 0 <= holdout_fraction <= 1, "holdout_fraction must be between 0 and 1"
        trn_nsamples = nsamples - int(nsamples * holdout_fraction)

        if holdout_shuffle:
            perm = np.random.choice(nsamples, size=nsamples, replace=with_replacement)
            #perm = np.random.permutation(nsamples)
            trn_indices = perm[:trn_nsamples]
            tst_indices = perm[trn_nsamples:]
        else:
            trn_indices = np.arange(trn_nsamples)
            tst_indices = np.arange(trn_nsamples, nsamples)
        return trn_indices, tst_indices

    def split_train_test(self, **holdout_options):
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
            return self.rev_g_samples.mean(axis=0)


    def g_cov(self):
        # Covariance of g_samples
        return torch.cov(self.g_samples.T, correction=0)
    
    
    def rev_g_cov(self):
        # Covariance of reverse samples
        if self.rev_g_samples is None: # antisymmetric observables, they have the same covariance matrix
            return self.g_cov()
        else:
            torch.cov(self.rev_g_samples.T, correction=0)    
    

    def split_train_test(self, **holdout_opts):
        # Split current data set into training and heldout testing part
        trn_indices, tst_indices = self._get_trn_indices(nsamples=self.forward_nsamples, **holdout_opts)

        if self.rev_g_samples is not None:
            if self.nsamples != self.forward_nsamples:
                trn_indices_rev, tst_indices_rev = self._get_trn_indices(nsamples=self.nsamples, **holdout_opts)
            else: # assume that forward and reverse samples are paired (there is the same number), 
                trn_indices_rev = trn_indices
                tst_indices_rev = tst_indices
            rev_trn = self.rev_g_samples[trn_indices_rev]
            rev_tst = self.rev_g_samples[tst_indices_rev]
        else:
            rev_trn, rev_tst = None, None 

        trn = self.__class__(g_samples=self.g_samples[trn_indices], rev_g_samples=rev_trn)
        tst = self.__class__(g_samples=self.g_samples[tst_indices], rev_g_samples=rev_tst)
        return trn, tst
    

    def get_random_batch(self, batch_size):
        indices = np.random.choice(self.nsamples, size=batch_size, replace=True)
        if self.rev_g_samples is not None:
            if self.nsamples != self.forward_nsamples:
                rev_indices = np.random.choice(self.forward_nsamples, size=batch_size, replace=True)
            else: # assume that forward and reverse samples are paired (there is the same number), 
                rev_indices = indices
            return self.__class__(g_samples=self.g_samples[indices], rev_g_samples=self.rev_g_samples[rev_indices])
        else:
            return self.__class__(g_samples=self.g_samples[indices])


    def get_tilted_statistics(self, theta, return_mean=False, return_covariance=False, return_objective=False, num_chunks=None):
        # Compute tilted statistics under the reverse distribution titled by theta. This may include
        # the objective ( θᵀ<g> - ln(<exp(θᵀg)>_{~p}) ), the tilted mean, and the tilted covariance

        assert return_mean or return_covariance or return_objective

        vals  = {}
        theta = numpy_to_torch(theta)

        use_rev = self.rev_g_samples is not None  # antisymmetric observables or not

        with torch.no_grad():
        
            th_g = self.rev_g_samples @ theta if use_rev else -(self.g_samples @ theta)

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
                mean = exp_tilt @ self.rev_g_samples if use_rev else -(exp_tilt @ self.g_samples)
                mean = mean / (self.nsamples * norm_const)
                vals['tilted_mean'] = mean

                if return_covariance:
                    if num_chunks is None:
                        if use_rev:
#                            K = torch.einsum('k,ki,kj->ij', exp_tilt, self.rev_g_samples, self.rev_g_samples)
                            weighted_rev_g = exp_tilt[:, None] * self.rev_g_samples  # shape: (k, i)
                            K = weighted_rev_g.T @ self.rev_g_samples
                        else:
#                            K = torch.einsum('k,ki,kj->ij', exp_tilt, self.g_samples, self.g_samples)
                            weighted_rev_g = exp_tilt[:, None] * self.g_samples
                            K = weighted_rev_g.T @ self.g_samples
                        vals['tilted_covariance'] = K / (self.nsamples * norm_const) - torch.outer(mean, mean)

                    elif num_chunks == -1:
                        if use_rev:
                            vals['tilted_covariance'] = torch.cov(self.rev_g_samples.T, correction=0, aweights=exp_tilt)
                        else:
                            vals['tilted_covariance'] = torch.cov(self.g_samples.T, correction=0, aweights=exp_tilt)

                    else:
                        # Chunked computation
                        K = torch.zeros((self.nobservables, self.nobservables), dtype=theta.dtype, device=self.device)
                        chunk_size = (self.nsamples + num_chunks - 1) // num_chunks  # Ceiling division

                        for r in range(num_chunks):
                            start = r * chunk_size
                            end = min((r + 1) * chunk_size, self.nsamples)
                            g_chunk = self.rev_g_samples[start:end] if use_rev else -self.g_samples[start:end]
                            
#                            K += torch.einsum('k,ki,kj->ij', exp_tilt[start:end], g_chunk, g_chunk)
                            weighted_g = exp_tilt[start:end][:, None] * g_chunk
                            K += weighted_g.T @ g_chunk

                        vals['tilted_covariance'] = K / (self.nsamples * norm_const) - torch.outer(mean, mean)

#                        if use_rev:
#                            K = torch.einsum('k,ki,kj->ij', exp_tilt, self.rev_g_samples, self.rev_g_samples)
#                        else:
#                            K = torch.einsum('k,ki,kj->ij', exp_tilt, self.g_samples, self.g_samples)

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


    def split_train_test(self, **holdout_opts):
        # Split current data set into training and heldout testing part
        trn_indices, tst_indices = self._get_trn_indices(self.nsamples, **holdout_opts)
        trn = type(self)(X0=self.X0[trn_indices], X1=self.X1[trn_indices])
        tst = type(self)(X0=self.X0[tst_indices], X1=self.X1[tst_indices])
        return trn, tst

    def get_random_batch(self, batch_size):
        indices = np.random.choice(self.nsamples, size=batch_size, replace=True)
        return self.__class__(X0=self.X0[indices], X1=self.X1[indices])

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
    def __init__(self, X0, X1):
        # Arguments:
        # X0          : 2d tensor (nsamples x N) containing initial states of the system
        # X1          : 2d tensor (nsamples x N) containing final states of the system
        
        assert(X0.shape == X1.shape)
        
        self.X0 = numpy_to_torch(X0)
        self.X1 = numpy_to_torch(X1)
        self.diffX = self.X1 - self.X0
        
        # self.N indicates dimensionality of the system
        self.nsamples, self.N = self.X0.shape
        self.nsamples = self.nsamples

        self.device = self.X0.device
    
        self.nobservables = self.g_mean.shape[0]

        # print(torch.any(self.diffX != 0, dim=1).to(torch.float32).mean())

    @functools.cached_property
    def g_mean(self): # Calculate mean of observables under forward process
        # Here we consider g_{ij} = (x_i' - x_i) x_j
        return (self.diffX.T @ self.X0 / self.nsamples).flatten()
    

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
        #theta2d.fill_diagonal_(0)  # Set diagonal to 0
        th_g    = -torch.einsum('ij,ki,kj->k', theta2d, self.diffX, self.X1)

        th_g_max = torch.max(th_g)
        exp_tilt = torch.exp(th_g - th_g_max)
        norm_const = torch.mean(exp_tilt)
        
        if return_objective:
            # To return 'true' normalization constant, we need to correct again for th_g_max
            log_Z = torch.log(norm_const) + th_g_max
            vals['objective'] = float(theta @ self.g_mean - log_Z)


        if return_mean:
            weights = exp_tilt / norm_const
            tilted_g_mean = -torch.einsum('k,ki,kj->ij', weights, self.diffX, self.X1)  / self.nsamples
            vals['tilted_mean'] = tilted_g_mean.flatten()

        return vals




# EPEstimators return Solution namedtuples such as the following
#   objective (float) : estimate of EP
#   theta (torch tensor of length nobservables) : optimal conjugate parameters
#   trn_objective (float) : estimate of EP on training data (if holdout is used)
Solution = namedtuple('Solution', ['objective', 'theta', 'trn_objective'], defaults=[None])

def _get_valid_solution(objective, theta, nsamples, trn_objective=None):
    # This returns a Solution object, after doing some basic sanity checking of the values
    # This checking is useful in the undersampled regime with many dimensions and few samles
    if objective < 0:
        # EP estimate should never be negative, as we can always achieve objective=0 with all 0s theta
        objective = 0.0 
        if theta is not None:
            theta = 0*theta
    elif objective >= np.log(nsamples):
        # EP estimate should not be larger than log(# samples), because it is not possible
        # to estimate KL divergence larger than log(m) from m samples
        objective = np.log(nsamples)
    return Solution(objective=objective, theta=theta, trn_objective=trn_objective)


# ==========================================
# Entropy production (EP) estimation methods 
# ==========================================
def get_EP_Newton(data, theta_init=None, verbose=0, holdout_data=None, 
                    max_iter=None, tol=1e-8, linsolve_eps=1e-4, num_chunks=None,
                    trust_radius=None, solve_constrained=True, adjust_radius=False,
                    eta0=0.0, eta1=0.25, eta2=0.75, trust_radius_min=1e-3, trust_radius_max=1000.0,
                    trust_radius_adjust_max_iter=100, patience=10):
    # Estimate EP by optimizing objective using Newton's method. We support advanced
    # features like Newton's method with trust region constraints
    #
    # Arguments (all optional):
    #   data             : Dataset object providing statistics of observables   
    #   theta_init       : torch tensor specifiying initial parameters (set to 0s if None)
    #   verbose (int)    : Verbosity level for printing debugging information (0=no printing)
    #   holdout_data     : if not None, we use this dataset for early stopping 
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

    if max_iter is None:
        max_iter = 1000

    with torch.no_grad():   # We don't need to torch to keep track of gradients (sometimes a bit faster)
    
            
        if holdout_data:
            f_cur_trn = f_cur_tst = f_new_trn = f_new_tst = 0.0
        else:
            f_cur_trn = f_new_trn = 0.0
        
        if theta_init is not None:
            theta = numpy_to_torch(theta_init).clone()
        else:
            theta = torch.zeros(data.nobservables, device=data.device)
        I     = torch.eye(len(theta), device=theta.device)
        
        if holdout_data:
            best_theta = theta.clone()
            best_val = -float('inf')
            no_improve_count = 0

        for t in range(max_iter):
            if verbose and verbose > 1: 
                print(f'{funcname} : iteration {t:5d} f_cur_trn={f_cur_trn: 3f}', 
                                                    f'f_cur_tst={f_cur_tst: 3f}' if holdout_data is not None else '')

            # Find Newton step direction. We first get gradient and Hessian
            stats = data.get_tilted_statistics(theta, return_mean=True, return_covariance=True, num_chunks=num_chunks)
            g_theta = stats['tilted_mean']
            H_theta = stats['tilted_covariance']

            if is_infnan(H_theta.sum()): 
                # Error occured, usually it means theta is too big
                if verbose: print(f'{funcname} : [Stopping] Invalid Hessian')
                break

            grad = data.g_mean - g_theta
            H_theta += linsolve_eps * I  # regularize Hessian by adding a small I

            if trust_radius is not None and solve_constrained:
                # Solve the constrained trust region problem using the
                # Steihaug-Toint Truncated Conjugate-Gradient Method

                for tr_iter in range(trust_radius_adjust_max_iter): # loop until trust_radius is adjusted properly
                    delta_theta = steihaug_toint_cg(A=H_theta, b=grad, trust_radius=trust_radius)
                    new_theta  = theta + delta_theta
                    f_new_trn  = data.get_objective(new_theta)

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
                            if verbose: print(f"{funcname} : trust_radius_min={trust_radius_min} reached!")
                            trust_radius = trust_radius_min
                            break

                        if rho < eta1:
                            trust_radius *= 0.25
                            if verbose > 1: print(f"{funcname} : reducing trust_radius to {trust_radius} in trust radius iteration={tr_iter}")
                        elif rho > eta2 and delta_theta.norm() >= trust_radius:
                            trust_radius = min(2.0 * trust_radius, trust_radius_max)
                            if verbose > 1: print(f"{funcname} : increasing trust_radius to {trust_radius} in trust radius iteration={tr_iter}")


                else: # for loop finished without breaking
                    if verbose: print(f"{funcname} : max_iter reached in adjust_radius loop!")

            else:
                # Find regular Newton direction
                delta_theta = solve_linear_psd(A=H_theta, b=grad)
                if trust_radius is not None:
                    # We do a quick-and-dirty approximation to trust-region constraint
                    # by rescaling norm of Newton step if it is too large
                    delta_theta *= trust_radius/max(trust_radius, delta_theta.norm())
                new_theta  = theta + delta_theta
                f_new_trn  = data.get_objective(new_theta)


            last_round = False # set to True if we want break after updating theta and objective values

#            if holdout_data is None:
#                if is_infnan(f_new_trn):  
#                    if verbose: print(f"{funcname} : [Stopping] Invalid value {f_new_trn} in training objective at iter {t}")
#                    break                  # Training value should be finite and increasing
#                elif f_new_trn <= f_cur_trn:
#                    if verbose: print(f"[Stopping] Training objective did not improve (f_new_trn <= f_cur_trn) at iter {t}")
#                    break
#                elif np.abs(f_new_trn - f_cur_trn) <= tol: 
#                    if verbose: print(f"{funcname} : [Converged] Train objective change below tol={tol} at iter {t}")
#                    last_round = True      # Break when training objective stops improving by more than tol
#                elif f_new_trn > np.log(data.nsamples):  
#                    # One cannot reliably estimate KL divergence larger than log(# samples).
#                    # This is a signature of undersampling; when it happens, we clip our estimate of the 
#                    # objective and exit
#                    print(f"{funcname} : [Clipping] Training objective exceeded log(#samples), clipping to log(nsamples) at iter {t}")
#                    f_new_trn = np.log(data.nsamples)
#                    last_round = True

#            else                # Do the same checks but now on the heldout test data
#                f_new_tst = holdout_data.get_objective(new_theta) 
#                if is_infnan(f_new_tst):
#                    if verbose: print(f"{funcname} : [Stopping] Invalid value {f_new_tst} in test objective at iter {t}")
#                    break
#                elif f_new_tst <= f_cur_tst:
#                    if verbose: print(f"[Stopping] Testing objective did not improve (f_new_tst <= f_cur_tst) at iter {t}")
#                    break
#                elif np.abs(f_new_tst - f_cur_tst) <= tol:
#                    if verbose: print(f"{funcname} : [Converged] Test objective change below tol={tol} at iter {t}")
#                    last_round = True
#                elif f_new_tst > np.log(holdout_data.nsamples):
#                    if verbose: print(f"{funcname} : [Clipping] Test objective exceeded log(#samples), clipping at iter {t}")
#                    f_new_tst = np.log(holdout_data.nsamples)
#                    last_round = True

            if is_infnan(f_new_trn):  
                if verbose: print(f"{funcname} : [Stopping] Invalid value {f_new_trn} in training objective at iter {t}")
                break                  # Training value should be finite and increasing
            elif np.abs(f_new_trn - f_cur_trn) < tol: 
                if verbose: print(f"{funcname} : [Converged] Train objective change below tol={tol} at iter {t}")
                last_round = True      # Break when training objective stops improving by more than tol
            elif f_new_trn > np.log(data.nsamples):  
                # One cannot reliably estimate KL divergence larger than log(# samples).
                # This is a signature of undersampling; when it happens, we clip our estimate of the 
                # objective and exit
                if verbose: print(f"{funcname} : [Clipping] Training objective exceeded log(#samples), clipping to log(nsamples) at iter {t}")
                f_new_trn = np.log(data.nsamples)
                last_round = True

            if holdout_data is not None:
                f_new_tst = holdout_data.get_objective(new_theta)
                if is_infnan(f_new_tst):
                    if verbose: print(f"{funcname} : [Stopping] Invalid value {f_new_tst} in test objective at iter {t}")
                    break
                elif f_new_tst > np.log(holdout_data.nsamples):
                    if verbose: print(f"{funcname} : [Clipping] Test objective exceeded log(#samples), clipping at iter {t}")
                    f_new_tst = np.log(holdout_data.nsamples)
                    last_round = True
                
                if f_new_tst > best_val:
                    best_val = f_new_tst
                    best_val_trn = f_new_trn
                    best_theta = new_theta.clone()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        last_round=True
                        if verbose: print(f"{funcname} : [Early stopping] No improvement for {patience} steps at iter {t}.")
                f_cur_tst = f_new_tst
                
            # Update our estimate of the paramters and objective value
            f_cur_trn, theta = f_new_trn, new_theta
            if holdout_data is not None:
                f_cur_tst = f_new_tst

            if last_round:
                if holdout_data is not None:
                    f_cur_tst = best_val
                    f_cur_trn = best_val_trn
                    theta = best_theta.clone()
                break

        else:   # for loop did not break
            if max_iter > 10 and verbose:
                # print warning about max iterations reached, but only if its a large number
                print(f'max_iter {max_iter} reached in get_EP_Newton!')
            pass

        if holdout_data is not None:
            return _get_valid_solution(objective=f_cur_tst, theta=theta, nsamples=data.nsamples, trn_objective=f_cur_trn)
        else:
            return _get_valid_solution(objective=f_cur_trn, theta=theta, nsamples=data.nsamples)



def get_EP_GradientAscent(data, theta_init=None, verbose=0, holdout_data=None, report_every=10,
                            max_iter=None, lr=None, patience=10, tol=1e-4, 
                            use_Adam=True, use_BB=False, beta1=0.9, beta2=0.999, eps=1e-8, skip_warm_up=False,
                            batch_size=None):
    # Estimate EP using gradient ascent algorithm

    # Arguments:
    #   data             : Dataset object providing statistics of observables   
    #   theta_init       : torch tensor specifiying initial parameters (set to 0s if None)
    #   verbose (int)    : Verbosity level for printing debugging information (0=no printing)
    #   holdout_data     : if not None, we use this dataset for early stopping 
    #   max_iter (int)   : maximum number of iterations
    #   lr               : learning rate
    #   tol (float)      : early stopping once objective does not improve by more than tol
    #   verbose (int)    : Verbosity level for printing debugging information (0=no printing)
    #   report_every (int) : if verbose > 1, we report every report_every iterations
    #
    # Adam-related arguments:
    #   use_Adam       : whether to use use_Adam algorithm w/ momentum or just regular grad. ascent
    #   beta1, beta2   : Adam moment decay parameters
    #   skip_warm_up   : Adam option
    #
    # Returns:
    #   Solution object with objective (EP estimate), theta, and trn_objective (if holdout)

    funcname = 'get_EP_GradientAscent'

    if max_iter is None:
        max_iter = 10000
    if lr is None:
        lr = 0.01
    
    if holdout_data is not None:
        f_cur_trn = f_cur_tst = f_new_trn = f_new_tst = np.nan
    else:
        f_cur_trn = f_new_tst = np.nan

    with torch.no_grad():  # We calculate our own gradients, so we don't need to torch to do it (sometimes a bit faster)

        if theta_init is not None:
            new_theta = numpy_to_torch(theta_init)
        else:
            new_theta = torch.zeros(data.nobservables, device=data.device)

        m = torch.zeros_like(new_theta)
        v = torch.zeros_like(new_theta)
        
        best_tst_score   = -float('inf')  # for maximization
        patience_counter = 0
        old_time         = time.time()

        old_grad = delta_theta = None
        for t in range(max_iter):
            if verbose and verbose > 1 and report_every > 0 and t % report_every == 0:
                new_time = time.time()
                print(f'{funcname} : iteration {t:5d} | {(new_time - old_time)/report_every:3f}s/iter | f_cur_trn={f_cur_trn: 3f}', 
                      f'f_cur_tst={f_cur_tst: 3f}' if holdout_data is not None else '')
                old_time = new_time


            if batch_size is not None:
                c_data = data.get_random_batch(batch_size)
            else:
                c_data = data

            tilted_stats = c_data.get_tilted_statistics(new_theta, return_objective=True, return_mean=True)

            grad         = c_data.g_mean - tilted_stats['tilted_mean']   # Gradient


            f_new_trn    = tilted_stats['objective']

            # Different conditions that will stop optimization. See get_EP_Newton above
            # for a description of different branches
            last_round = False # flag that indicates whether to break after updating values
            if is_infnan(f_new_trn):
                if verbose: print(f"{funcname} : [Stopping] Invalid value {f_new_trn} in training objective at iter {t}")
                break
#                elif f_new_trn <= f_cur_trn and t > min_iter:
#                    print(f"[Stopping] Training objective did not improve (f_new_trn <= f_cur_trn) at iter {t}")
#                    break
            elif f_new_trn > np.log(data.nsamples):
                if verbose: print(f"{funcname} : [Clipping] Training objective exceeded log(samples), clipping to log(nsamples) at iter {t}")
                f_new_trn = np.log(data.nsamples)
                last_round = True
            elif np.abs(f_new_trn - f_cur_trn) < tol:
                if verbose: print(f"{funcname} : [Converged] Training objective change below tol={tol} at iter {t}")
                last_round = True

            if holdout_data is not None: #  and t > 150:
                f_new_tst = holdout_data.get_objective(new_theta) 
                #train_gain = f_new_trn - f_cur_trn
                #test_drop = f_cur_tst - f_new_tst
                if is_infnan(f_new_tst):
                    if verbose: print(f"{funcname} : [Stopping] Invalid value {f_new_tst} in test objective at iter {t}")
                    break
#                    elif test_drop > 0 and train_gain > tol :
#                        print(f"[Stopping] Test objective did not improve (f_new_tst <= f_cur_tst and (f_new_trn - f_cur_trn) > tol ) at iter {t}")
#                        break
                elif f_new_tst > np.log(holdout_data.nsamples):
                    if verbose: print(f"{funcname} : [Clipping] Test objective exceeded log(#samples), clipping at iter {t}")
                    f_new_tst = np.log(holdout_data.nsamples)
                    last_round = True
                #elif np.abs(f_new_tst - f_cur_tst) < tol:
                #    if verbose: print(f"{funcname} : [Converged] Test objective change below tol={tol} at iter {t}")
                #    last_round = True
                
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
            if holdout_data is not None:
                f_cur_tst = f_new_tst

            if last_round:
                break


            if use_BB and delta_theta is not None and old_grad is not None:
                # Barzilai-Borwein method, short-step version
                d_grad  = grad - old_grad
                c_alpha = -(delta_theta @ d_grad) / (d_grad @ d_grad)
                # print(t, 'here', c_alpha, grad)

                delta_theta = c_alpha * grad 



            elif not use_BB and use_Adam:
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
                delta_theta = lr * m_hat / (v_hat.sqrt() + eps)

            else:
                # regular gradient ascent
                delta_theta = lr * grad

            old_grad = grad

            new_theta = theta + delta_theta


        else:   # for loop did not break
            if verbose:
                # print warning about max iterations reached, but only if its a large number
                print(f'max_iter {max_iter} reached in get_EP_GradientAscent!')
            pass

        if holdout_data is not None:
            return _get_valid_solution(objective=f_cur_tst, theta=theta, nsamples=data.nsamples, trn_objective=f_cur_trn)
        else:
            return _get_valid_solution(objective=f_cur_trn, theta=theta, nsamples=data.nsamples)



# Estimate EP using the multidimensional TUR method
def get_EP_MTUR(data, num_chunks=None, linsolve_eps=1e-4):
    # The MTUR is defined as (1/2) (<g>_(p - ~p))^T K^-1 (<g>_p - <g>_(p - ~p))
    # where μ = (p + ~p)/2 is the mixture of the forward and reverse distributions
    # and K^-1 is covariance matrix of g under (p + ~p)/2.
    # 
    # Optional arguments
    #   num_chunks (int)         : chunk covariance computations to reduce memory requirements
    #   linsolve_eps (float)     : regularization parameter for covariance matrices, used to improve
    #                               numerical stability of linear solvers

    mean_diff      = data.g_mean - data.rev_g_mean

    combined_mean  = (data.g_mean + data.rev_g_mean)/2

    # now compute covariance of (p + ~p)/2
    if num_chunks == -1:
        second_moment_forward = data.g_cov() + torch.outer(data.g_mean, data.g_mean)
        second_moment_reverse = data.rev_g_cov() + torch.outer(data.rev_g_mean, data.rev_g_mean)
        combined_cov = (second_moment_forward + second_moment_reverse)/2 - torch.outer(combined_mean, combined_mean)
    
    else:
        if data.rev_g_samples is None:
            combined_samples = data.g_samples
            num_total        = combined_samples.shape[0]
            weights          = torch.empty(num_total, device=data.device)
            weights[:data.nsamples] = 1.0/data.nsamples
        else:
            combined_samples = torch.concatenate([data.g_samples, data.rev_g_samples], dim=0)
            num_total        = combined_samples.shape[0]
            weights          = torch.empty(num_total, device=data.device)
            weights[:data.nsamples] = 1.0/data.nsamples/2.0
            weights[data.nsamples:] = 1.0/data.nsamples/2.0


        if num_chunks is None:
#            combined_cov = torch.einsum('k,ki,kj->ij', weights, combined_samples, combined_samples)
            weighted_samples = weights[:, None] * combined_samples
            combined_cov = weighted_samples.T @ combined_samples 
        else:
            # Chunked computation
            combined_cov = torch.zeros((data.nobservables, data.nobservables), device=data.device)
            chunk_size = (num_total + num_chunks - 1) // num_chunks  # Ceiling division

            for r in range(num_chunks):
                start = r * chunk_size
                end = min((r + 1) * chunk_size, num_total)
                chunk = combined_samples[start:end]
                weighted_samples = weights[start:end,None] * chunk
                combined_cov +=  weighted_samples.T @ chunk 
#                combined_cov += torch.einsum('k,ki,kj->ij', weights[start:end], chunk, chunk)

    x  = solve_linear_psd(combined_cov + linsolve_eps*eye_like(combined_cov), mean_diff)
    objective = float(x @ mean_diff)/2

    return _get_valid_solution(objective=objective, theta=None, nsamples=data.nsamples)

