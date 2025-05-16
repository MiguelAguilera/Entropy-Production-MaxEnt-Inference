# This file contains code to calculate observables and define data-based objectives

import os, functools
import numpy as np

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]='1'
import torch

from utils import numpy_to_torch
import optimizers

# The following functions are used to calculate observables for the nonequilibrium spin model
# They take as input the states of the system and the indices of the spins that flipped,
# as produced by the Monte Carlo simulation in spin_model.py
def get_g_observables(S, F, i):
    # Given states S ∈ {-1,1} in which spin i flipped, we calculate the observables 
    #        g_{ij}(x) = (x_i' - x_i) x_j 
    # for all j, where x_i' and x_i indicate the state of spin i after and before the jump, 
    # and x_j is the state of every other spin.
    # Here F indicates which spins flipped. We try to use GPU and in-place operations
    # where possible, to increase speed while reducing GPU memory requirements
    S_i      = S[F[:,i],:]
    X        = numpy_to_torch(np.delete(S_i, i, axis=1))
    y        = numpy_to_torch(S_i[:,i])
    X.mul_( (-2 * y).unsqueeze(1) )  # in place multiplication
    return X.contiguous()


# Same as get_g_observables( ... ), but we take binary states S_bin ∈ {0,1} and convert {0,1}→{−1,1}
# This can be faster, since we can sometimes do the conversion on the GPU
def get_g_observables_bin(S_bin, F, i):
    S_bin_i  = S_bin[F[:,i],:]
    X        = numpy_to_torch(np.delete(S_bin_i, i, axis=1))*2-1
    y        = numpy_to_torch(S_bin_i[:,i])*2-1
    X.mul_( (-2 * y).unsqueeze(1) )  # in place multiplication
    return X.contiguous()

    # The following can be a bit faster, but may use double the memory on the GPU
    # S_i  = numpy_to_torch(S_bin[F[:,i],:])*2-1
    # y    = -2 * S_i[:, i]
    # S_i  = torch.hstack((S_i[:,:i], S_i[:,i+1:]))
    # S_i.mul_( y.unsqueeze(1) )
    # return S_i.contiguous()


# The following classes are used to define the optimizers.Objective classes
# based on samples of observables or state transitions. 

# The DatasetBase class is the base class for all data-based objectives
class DatasetBase(optimizers.Objective):

    # We should implement the following methods
    @functools.cached_property
    def g_mean(self)     : raise NotImplementedError 

    @functools.cached_property
    def rev_g_mean(self): raise NotImplementedError 
    def g_cov(self)     : raise NotImplementedError 
    def rev_g_cov(self) : raise NotImplementedError 

    def get_tilted_statistics(self, theta, **kwargs):
        raise NotImplementedError("get_tilted_statistics is not implemented")
    
    def get_objective(self, theta):
        # Return objective value for parameters theta
        v = self.get_tilted_statistics(theta, return_objective=True)
        return v['objective']


    def get_gradient(self, theta):
        stats = self.get_tilted_statistics(theta, return_mean=True)
        g_theta = stats['tilted_mean']
        return self.g_mean - g_theta
    

    def get_hessian(self, theta):
        stats = self.get_tilted_statistics(theta, return_covariance=True)
        g_cov = stats['tilted_covariance']
        return -g_cov


    def _get_null_statistics(self, theta, return_mean=False, return_covariance=False, return_objective=False):
        nan = float('nan')
        d = {'tilted_mean': theta * nan, 'objective': nan}
        if return_covariance:
            d['tilted_covariance'] = torch.zeros((self.nobservables, self.nobservables), dtype=theta.dtype, device=theta.device)*nan
        return d
    

    # This method is used to get the indices for training, validation, and test sets
    @staticmethod
    def get_trn_val_tst_indices(nsamples, val_fraction=0.1, test_fraction=0.1, shuffle=True, with_replacement=False):
        # Get indices for training and testing data
        assert 0 <= test_fraction <= 1, "test_fraction must be between 0 and 1"
        trn_nsamples = int(nsamples * (1 - val_fraction - test_fraction))
        val_nsamples = int(nsamples * val_fraction)

        if shuffle:
            perm = np.random.choice(nsamples, size=nsamples, replace=with_replacement)
            trn_indices = perm[:trn_nsamples]
            val_indices = perm[trn_nsamples:trn_nsamples+val_nsamples]
            tst_indices = perm[trn_nsamples+val_nsamples:]
        else:
            trn_indices = np.arange(trn_nsamples)
            val_indices = np.arange(trn_nsamples, trn_nsamples+val_nsamples)
            tst_indices = np.arange(trn_nsamples+val_nsamples, nsamples)
        return trn_indices, val_indices, tst_indices

    # This method is used to split the dataset into training and testing sets
    def split_train_test(self, test_fraction=0.1, **kw_options):
        tst, _, trn = self.split_train_val_test(test_fraction=test_fraction, **kw_options)
        return trn, tst

    # This method is used to split the dataset into training, validation, and testing sets
    def split_train_val_test(self, val_fraction=0.1, test_fraction=0.1, shuffle=True):
        raise NotImplementedError("split_train_val_test is not implemented")


# TODO: Cache intermediate results

# This class implements an Objective using samples of observables
class Dataset(DatasetBase):
    def __init__(self, g_samples, rev_g_samples=None, num_chunks=None):
        # Arguments:
        #   g_samples                : 2d tensor (nsamples x nobservables) containing samples of observables
        #                              under reverse process 
        #   rev_g_samples            : 2d tensor (nsamples x nobservables) containing samples of observables
        #                              under reverse process . If None, we assume antisymmetric observables
        #                              where rev_g_samples = -g_samples
        #   num_chunks               : number of chunks to use for computing tilted statistics. This is useful
        #                              for large datasets, where we want to reduce memory usage.

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

        if num_chunks is not None:
            assert(num_chunks > 0)
            if num_chunks > self.nsamples:
                num_chunks = self.nsamples 
        self.num_chunks = num_chunks


    @functools.cached_property
    def g_mean(self):
        return self.g_samples.mean(axis=0)
    
    def g_cov(self): # Covariance of g_samples
        return torch.cov(self.g_samples.T, correction=0)
    
    def rev_g_cov(self):
        # Covariance of reverse samples
        if self.rev_g_samples is None: # antisymmetric observables, so they have the same covariance matrix
            return self.g_cov()
        else:
            return torch.cov(self.rev_g_samples.T, correction=0)

    @functools.cached_property
    def rev_g_mean(self):
        if self.rev_g_samples is None: # antisymmetric observables
            return -self.g_mean
        else:
            return self.rev_g_samples.mean(axis=0)

    
    def split_train_val_test(self, **split_opts):  # Split current data set into training, validation, and testing part
        trn_indices, val_indices, tst_indices = self.get_trn_val_tst_indices(nsamples=self.forward_nsamples, **split_opts)

        if self.rev_g_samples is not None:
            if self.nsamples != self.forward_nsamples:
                trn_indices_rev, val_indices_rev, tst_indices_rev = self.get_trn_val_tst_indices(nsamples=self.nsamples, **split_opts)
            else: # assume that forward and reverse samples are paired (there is the same number), 
                trn_indices_rev = trn_indices
                val_indices_rev = val_indices
                tst_indices_rev = tst_indices
            rev_trn = self.rev_g_samples[trn_indices_rev]
            rev_val = self.rev_g_samples[val_indices_rev]
            rev_tst = self.rev_g_samples[tst_indices_rev]
        else:
            rev_trn, rev_val, rev_tst = None, None, None

        trn = self.__class__(g_samples=self.g_samples[trn_indices], rev_g_samples=rev_trn)
        val = self.__class__(g_samples=self.g_samples[val_indices], rev_g_samples=rev_val)
        tst = self.__class__(g_samples=self.g_samples[tst_indices], rev_g_samples=rev_tst)

        return trn, val, tst


    def get_tilted_statistics(self, theta, return_mean=False, return_covariance=False, return_objective=False):
        # Compute tilted statistics under the reverse distribution titled by theta. This may include
        # the objective ( θᵀ<g> - ln(<exp(θᵀg)>_{~p}) ), the tilted mean, and the tilted covariance

        assert return_mean or return_covariance or return_objective

        if self.nsamples == 0:  # No observables
            return self._get_null_statistics(theta, return_mean, return_covariance, return_objective)

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
                    num_chunks = self.num_chunks
                    if num_chunks is None:
                        if use_rev:
                            weighted_rev_g = exp_tilt[:, None] * self.rev_g_samples  # shape: (k, i)
                            K = weighted_rev_g.T @ self.rev_g_samples
                        else:
                            weighted_rev_g = exp_tilt[:, None] * self.g_samples
                            K = weighted_rev_g.T @ self.g_samples
                        vals['tilted_covariance'] = K / (self.nsamples * norm_const) - torch.outer(mean, mean)

                    else:
                        # Chunked computation, helps reduce memory usage for large datasets
                        K = torch.zeros((self.nobservables, self.nobservables), dtype=theta.dtype, device=self.device)
                        chunk_size = (self.nsamples + num_chunks - 1) // num_chunks  # Ceiling division

                        for r in range(num_chunks):
                            start = r * chunk_size
                            end = min((r + 1) * chunk_size, self.nsamples)
                            g_chunk = self.rev_g_samples[start:end] if use_rev else -self.g_samples[start:end]
                            
                            weighted_g = exp_tilt[start:end][:, None] * g_chunk
                            K += weighted_g.T @ g_chunk

                        vals['tilted_covariance'] = K / (self.nsamples * norm_const) - torch.outer(mean, mean)

        return vals


    
# This class implements an Objective using samples of state transitions. We assume antisymmetric observables
class DatasetStateSamplesBase(DatasetBase):
    def __init__(self, X0, X1):
        # Arguments:
        # X0          : 2d tensor (nsamples x N) containing initial states of the system
        # X1          : 2d tensor (nsamples x N) containing final states of the system
        assert(X0.shape == X1.shape)
        self.X0 = numpy_to_torch(X0)
        self.X1 = numpy_to_torch(X1)
        self.nsamples, self.N = self.X0.shape             # self.N indicates dimensionality of the system
        self.device = self.X0.device
        self.nobservables = self.g_mean.shape[0]


    @functools.cached_property
    def rev_g_mean(self): # Calculate mean of observables under reverse process
        return -self.g_mean  # Antisymmetric observables


    def split_train_val_test(self, **split_opts):
        # Split current data set into training and heldout testing part
        trn_indices, val_indices, tst_indices = self.get_trn_val_tst_indices(self.nsamples, **split_opts)
        trn = type(self)(X0=self.X0[trn_indices], X1=self.X1[trn_indices])
        val = type(self)(X0=self.X0[val_indices], X1=self.X1[val_indices])
        tst = type(self)(X0=self.X0[tst_indices], X1=self.X1[tst_indices])
        return trn, val, tst




class CrossCorrelations1(DatasetStateSamplesBase):
    # This class is used to calculate the antisymmetric observables g_{ij} = x_i * x'_j - x'_i * x_j
    # without materializing the full g_samples matrix. 
    @functools.cached_property
    def g_mean(self): # Calculate mean of observables under forward process
        triu_indices = torch.triu_indices(self.N, self.N, offset=1, device=self.device)
        g_mean_raw = (self.X1.T @ self.X0 - self.X0.T @ self.X1 ) / self.nsamples  # shape (N, N)
        return g_mean_raw[triu_indices[0], triu_indices[1]]


    def get_tilted_statistics(self, theta, return_mean=False, return_covariance=False, return_objective=False):
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
        Theta = torch.zeros((self.N, self.N), dtype=theta.dtype, device=self.device)
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

    
class CrossCorrelations2(CrossCorrelations1):
    # This class is used to calculate the antisymmetric observables g_{ij} = (x'_i - x_i) x_j
    # without materializing the full g_samples matrix. 
    @functools.cached_property
    def diffX(self):
        # Calculate the difference between the final and initial states\
        return self.X1 - self.X0

    @functools.cached_property
    def g_mean(self): # Calculate mean of observables under forward process
        # Here we consider g_{ij} = (x_i' - x_i) x_j
        return (self.diffX.T @ self.X0 / self.nsamples).flatten()

    def get_tilted_statistics(self, theta, return_mean=False, return_covariance=False, return_objective=False):
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