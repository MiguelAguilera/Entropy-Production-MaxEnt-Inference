# ============================================================
# Entropy production (EP) estimation methods
# ============================================================

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

# TODO: Cache intermediate results
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
    
    def _get_null_statistics(self, theta, return_mean=False, return_covariance=False, return_objective=False):
        nan = float('nan')
        d = {'tilted_mean': theta * nan, 'objective': nan}
        if return_covariance:
            d['tilted_covariance'] = torch.zeros((self.nobservables, self.nobservables))*nan
        return d
    
    def get_objective(self, theta):
        # Return objective value for parameters theta
        v = self.get_tilted_statistics(theta, return_objective=True)
        return v['objective']


    @staticmethod
    def _get_trn_indices(nsamples, test_fraction=1/2, shuffle=True, with_replacement=False):
        # Get indices for training and testing data
        assert 0 <= test_fraction <= 1, "test_fraction must be between 0 and 1"
        trn_nsamples = nsamples - int(nsamples * test_fraction)

        if shuffle:
            perm = np.random.choice(nsamples, size=nsamples, replace=with_replacement)
            #perm = np.random.permutation(nsamples)
            trn_indices = perm[:trn_nsamples]
            tst_indices = perm[trn_nsamples:]
        else:
            trn_indices = np.arange(trn_nsamples)
            tst_indices = np.arange(trn_nsamples, nsamples)
        return trn_indices, tst_indices

    def split_train_test(self, **kw_options):
        raise NotImplementedError("split_train_test is not implemented")

    
    def split_train_val_test(self, val_fraction=0.1, test_fraction=0.1, shuffle=True):
        # Split dataset into train/val/test sets with default 80/10/10 split
        assert val_fraction + test_fraction < 1.0
        trn, val_tst = self.split_train_test(test_fraction=val_fraction + test_fraction, shuffle=shuffle)
        val, tst = val_tst.split_train_test(test_fraction=test_fraction / (val_fraction + test_fraction), shuffle=shuffle)
        return trn, val, tst


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

    def get_gradient(self, theta):
        stats = self.get_tilted_statistics(theta, return_mean=True)
        g_theta = stats['tilted_mean']
        return self.g_mean - g_theta
    

    def get_hessian(self, theta):
        stats = self.get_tilted_statistics(theta, return_covariance=True)
        g_cov = stats['tilted_covariance']
        return -g_cov

    def g_cov(self):
        # Covariance of g_samples
        return torch.cov(self.g_samples.T, correction=0)
    
    
    def rev_g_cov(self):
        # Covariance of reverse samples
        if self.rev_g_samples is None: # antisymmetric observables, they have the same covariance matrix
            return self.g_cov()
        else:
            torch.cov(self.rev_g_samples.T, correction=0)    
    

    def split_train_test(self, **split_opts):
        # Split current data set into training and heldout testing part
        trn_indices, tst_indices = self._get_trn_indices(nsamples=self.forward_nsamples, **split_opts)

        if self.rev_g_samples is not None:
            if self.nsamples != self.forward_nsamples:
                trn_indices_rev, tst_indices_rev = self._get_trn_indices(nsamples=self.nsamples, **split_opts)
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

        if self.nsamples == 0:  # No observables
            return self._get_null_statistics(theta, return_mean, return_covariance, return_objective)
        
        

        vals  = {}
        theta = numpy_to_torch(theta)

        use_rev = self.rev_g_samples is not None  # antisymmetric observables or not

        with torch.no_grad():
            
            # print((self.g_samples @ theta).sum())
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


    def split_train_test(self, **split_opts):
        # Split current data set into training and heldout testing part
        trn_indices, tst_indices = self._get_trn_indices(self.nsamples, **split_opts)
        trn = type(self)(X0=self.X0[trn_indices], X1=self.X1[trn_indices])
        tst = type(self)(X0=self.X0[tst_indices], X1=self.X1[tst_indices])
        return trn, tst
        
    def split_train_val_test(self, val_fraction=0.1, test_fraction=0.1, shuffle=True):
        # Split current data set into training, validation, and test parts
        assert val_fraction + test_fraction < 1.0

        if shuffle:
            perm = np.random.permutation(self.nsamples)
        else:
            perm = np.arange(self.nsamples)

        n_val = int(self.nsamples * val_fraction)
        n_test = int(self.nsamples * test_fraction)
        n_train = self.nsamples - n_val - n_test

        trn_indices = perm[:n_train]
        val_indices = perm[n_train:n_train + n_val]
        tst_indices = perm[n_train + n_val:]

        trn = type(self)(X0=self.X0[trn_indices], X1=self.X1[trn_indices])
        val = type(self)(X0=self.X0[val_indices], X1=self.X1[val_indices])
        tst = type(self)(X0=self.X0[tst_indices], X1=self.X1[tst_indices])
        return trn, val, tst

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
        
        if self.nsamples == 0:  # No observables
            return self._get_null_statistics(theta, return_mean, return_covariance, return_objective)
        
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

        if self.nsamples == 0:  # No observables
            return self._get_null_statistics(theta, return_mean, return_covariance, return_objective)
        
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


# ============================================================

import optimizers
def get_EP_Estimate(data, validation_data=None, test_data=None, verbose=0,
                    optimize_kwargs=None):

    x0 = torch.zeros(data.nobservables, device=data.device)
    if data.nsamples == 0 or \
        (validation_data is not None and validation_data.nsamples == 0) or \
        (test_data       is not None and test_data.nsamples       == 0):
        # There is not enough data to estimate the objective
        return 0, x0
    
    if optimize_kwargs is None:
        optimize_kwargs = {}
    
    o = optimizers.optimize(x0=x0,
                            objective=data, 
                            validation_objective=validation_data,
                            test_objective=test_data,
                            minimize=False,
                            **optimize_kwargs,
                            )
    
    return o.objective, o.x 

# TODO: Explain
def get_EP_Newton1Step(data, validation_data=None, test_data=None, verbose=0,
                    optimize_kwargs=None):
    if optimize_kwargs is None:
        optimize_kwargs = {}
    optimize_kwargs['max_iter'] = 1
    optimize_kwargs['optimizer'] = 'NewtonMethod'
    return get_EP_Estimate(data, validation_data=validation_data, test_data=test_data,
                           optimize_kwargs=optimize_kwargs)



def get_EP_MTUR(data, linsolve_eps=1e-4):
    # Estimate EP using the multidimensional TUR method
    # The MTUR is defined as (1/2) (<g>_(p - ~p))^T K^-1 (<g>_p - <g>_(p - ~p))
    # where μ = (p + ~p)/2 is the mixture of the forward and reverse distributions
    # and K^-1 is covariance matrix of g under (p + ~p)/2.
    # 
    # Optional arguments
    #   linsolve_eps (float)     : regularization parameter for covariance matrices, used to improve
    #                               numerical stability of linear solvers

    if data.nsamples == 0:  # There is not enough data to estimate the objective
        return _get_null_solution(data)

    mean_diff      = data.g_mean - data.rev_g_mean

    combined_mean  = (data.g_mean + data.rev_g_mean)/2

    # Compute covariance of (p + ~p)/2
    second_moment_forward = data.g_cov() + torch.outer(data.g_mean, data.g_mean)
    second_moment_reverse = data.rev_g_cov() + torch.outer(data.rev_g_mean, data.rev_g_mean)
    combined_cov = (second_moment_forward + second_moment_reverse)/2 - torch.outer(combined_mean, combined_mean)
    
    x  = solve_linear_psd(combined_cov + linsolve_eps*eye_like(combined_cov), mean_diff)
    objective = float(x @ mean_diff)/2

    return objective, x

