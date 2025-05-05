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
import scipy

from collections import namedtuple

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]='1'
import torch

from utils import *

# EP estimators return Solution namedtuples such as the following
#   objective (float) : estimate of EP
#   theta (torch tensor of length nobservables) : optimal conjugate parameters
#   tst_objective (float) : estimate of EP on heldout test data (if holdout is used)
Solution = namedtuple('Solution', ['objective', 'theta', 'tst_objective'], defaults=[None])

def numpy_to_torch(X):
    return torch.from_numpy(X.astype('float32')).to(torch.get_default_device()).contiguous()

class EPEstimators(object):
    def __init__(self, g_mean, rev_g_samples, g_mean_f_b=None, g_covariance_f_b=None, holdout_fraction=0.5, holdout_shuffle=False, num_chunks=None, linsolve_eps=1e-4):
        # Arguments:
        #   g_samples                : 1d tensor (1 x nobservables) containing means of observables of interest
        #                              under forward process
        #   rev_g_samples            : 2d tensor (nsamples x nobservables) containing samples of observables
        #                              under reverse process
        #   holdout_fraction (float) : fraction of samples to use as holdout test dataset (if holdout is used)
        #   holdout_shuffle (bool)   : whether to shuffle train/holdout assignments (if holdout is used) 
        #   num_chunks (int)         : chunk covariance computations to reduce memory requirements
        #   linsolve_eps (float)     : regularization parameter for covariance matrices, used to improve
        #                               numerical stability of linear solvers

        if not isinstance(g_mean, torch.Tensor): # Conver to torch tensor if needed
            if isinstance(g_mean, np.ndarray):
                g_mean = numpy_to_torch(g_mean)
            else:
                raise Exception("g_mean must be a torch tensor or numpy array")

        if not isinstance(rev_g_samples, torch.Tensor): # Conver to torch tensor if needed
            if isinstance(rev_g_samples, np.ndarray):
                rev_g_samples = numpy_to_torch(rev_g_samples)
            else:
                raise Exception("rev_g_samples must be a torch tensor or numpy array")


        self.g_mean           = g_mean
        self.g_mean_f_b       = g_mean_f_b
        self.g_covariance_f_b = g_covariance_f_b
        self.rev_g_samples    = rev_g_samples
        self.nsamples, self.nobservables = rev_g_samples.shape
        self.device           = rev_g_samples.device

        assert holdout_fraction is None or (0 <= holdout_fraction <= 1)
        self.holdout_fraction = holdout_fraction
        self.holdout_shuffle  = holdout_shuffle

        self.num_chunks       = num_chunks

        assert linsolve_eps  >= 0
        self.linsolve_eps     = linsolve_eps
            
        # Used for copying of object
        self._init_args = ['g_mean', 'holdout_fraction', 'holdout_shuffle', 'num_chunks', 'linsolve_eps']

        
    def split_train_test(self):
        # Split current data set into training part and heldout testing part
        if not hasattr(self, 'trn_tst_split_'): # training and testign splits are cached
            if self.holdout_shuffle:
                perm = np.random.permutation(self.nsamples)
                rev_g_samples = self.rev_g_samples[perm]
            else:
                rev_g_samples = self.rev_g_samples

            trn_nsamples = self.nsamples - int(self.nsamples*self.holdout_fraction)

            kw_args = {k : getattr(self,k) for k in self._init_args}
            trn = EPEstimators(rev_g_samples=rev_g_samples[:trn_nsamples,:], **kw_args)
            tst = EPEstimators(rev_g_samples=rev_g_samples[trn_nsamples:,:], **kw_args)
            self.trn_tst_split_ = trn, tst

        return self.trn_tst_split_

    # ====================================================================================
    # Methods to compute observable statistics under forward distribution
    # ====================================================================================
    # def g_mean(self):
    #     # Mean of g under forward process
    #     return self.g_mean

    # # def g_secondmoments(self):
    #     # Compute matrix of second moments of g observables
    #     if not hasattr(self, 'g_secondmoments_'): # Cache values for speed
    #         self.g_secondmoments_ = batch_outer(K, self.num_chunks)
    #     return self.g_secondmoments_

    # Compute outer product X @ X.T/nrow. Do it in batches if requested for lower memory requirements
    # nrow, ncol = X.shape
    # if num_chunks is None:
    #     K = X@X.T
    # else:
    #     # Chunked computation, sometimes helpful for memory reasons
    #     K = torch.zeros((ncol, ncol), device=X.device)
    #     chunk_size = (nrow + num_chunks - 1) // num_chunks  # Ceiling division

    #     for r in range(num_chunks):
    #         start = r * chunk_size
    #         end = min((r + 1) * chunk_size, ncol)
    #         g_chunk = X[start:end]
    #         K += g_chunk @ g_chunk.T
    # return K/nrow




    # def g_covariance(self): 
    #     # Compute covariance matrix of g observables
    #     if not hasattr(self, 'g_covariance_'): # Cache for speed
    #         self.g_covariance_ = self.g_secondmoments() - torch.outer(self.g_mean(), self.g_mean())
    #     return self.g_covariance_

    # ====================================================================================
    # Methods to compute observable statistics under tilted reverse distribution
    # ====================================================================================

    def _get_tilted_statistics(self, theta, return_mean=False, return_covariance=False, return_objective=False):
        # This internal method computes tilted statistics reverse distribution titled by theta. This may include
        # the objective ( θᵀ<g> - ln(<exp(θᵀg)>_{~p}) ), the tilted mean, and the tilted covariance

        assert return_mean or return_covariance or return_objective

        vals       = {}

        th_g       = self.rev_g_samples @ theta  # backward samples?

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


    def get_EP_MTUR(self):
        # Estimate EP using the multidimensional TUR method

        # The MTUR is defined as (1/2) (<g>_(p - ~p))^T K^-1 (<g>_p - <g>_(p - ~p))
        # where μ = (p + ~p)/2 is the mixture of the forward and reverse distributions
        # and K^-1 is covariance matrix of g under (p + ~p)/2.
        #
        # In our case, g is antisymmetric (<g>_p=-<g>_~p), so
        #        <g>_(p - ~p)  = 2<g>_p 
        #        [K^-1]_ij     = <g_i g_j>

        # # The code commented out below was for doing a 'heldout' estimate of the TUR
        # # For simplicity, we removed it 
        # if holdout:
        #     trn, tst  = self.split_train_test()
        #     sol       = trn.get_EP_MTUR(holdout=False)
        #     theta     = sol.theta
        #     tst_objective = tst.get_objective(theta)
        # else:
        #     A         = self.g_secondmoments()
        #     A        += self.linsolve_eps*eye_like(A)
        #     theta     = solve_linear_psd(A, 2*self.g_mean)
        #     tst_objective = None

        A = self.g_covariance_f_b + self.linsolve_eps*eye_like(self.g_covariance_f_b)
        theta     = solve_linear_psd(A, 2*self.g_mean_f_b)
        sigma     = float(theta @ self.g_mean)
        return self.get_valid_solution(objective=sigma, theta=theta)


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



    def get_EP_GradientAscent(self, theta_init=None, holdout=False, lr=0.01, max_iter=1000, tol=1e-4, verbose=False,
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
            if is_infnan(f_new_trn) or f_new_trn <= f_cur_trn:
                break
            elif f_new_trn > np.log(trn.nsamples):
                f_new_trn = np.log(trn.nsamples)
                last_round = True
            elif np.abs(f_new_trn - f_cur_trn) <= tol: 
                last_round = True

            if holdout:
                f_new_tst = tst.get_objective(new_theta) 
                if is_infnan(f_new_tst) or f_new_tst <= f_cur_tst:
                    break
                elif f_new_tst > np.log(tst.nsamples):
                    f_new_tst = np.log(tst.nsamples)
                    last_round = True
                elif np.abs(f_new_tst - f_cur_tst) <= tol:
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

