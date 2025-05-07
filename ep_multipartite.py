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
#   theta (torch tensor of length N-1) : optimal parameters
#   tst_objective (float) : estimate of EP on heldout test data (if holdout is used)
Solution = namedtuple('Solution', ['objective', 'theta', 'tst_objective'], defaults=[None])

def add_i(theta, i):
    return torch.cat([theta[:i], torch.tensor([0.0], dtype=theta.dtype, device=theta.device), theta[i:]])

def remove_i_rowcol(K, i):
    device = K.device
    # Remove i-th row
    K_reduced = torch.cat([K[:i], K[i+1:]], dim=0)
    # Remove i-th column
    K_reduced = torch.cat([K_reduced[:, :i], K_reduced[:, i+1:]], dim=1)
    return K_reduced.to(device)

class EPEstimators(object):
    def __init__(self, S, i, holdout_fraction=0.5, holdout_shuffle=False, num_chunks=None, linsolve_eps=1e-4):
        # Arguments:
        #   S (torch tensor)         : 2d tensor (nflips x nspins) containing samples of 
        #                              states of the system in which spin i changed state
        #   i (int)                  : index of spin which changed state
        #   holdout_fraction (float) : fraction of samples to use as holdout test dataset (if holdout is used)
        #   holdout_shuffle (bool)   : whether to shuffle train/holdout assignments (if holdout is used) 
        #   num_chunks (int)         : chunk covariance computations to reduce memory requirements
        #   linsolve_eps (float)     : regularization parameter for covariance matrices, used to improve
        #                              numerical stability of linear solvers

        if not isinstance(S, torch.Tensor): # Conver to torch tensor if needed
            if isinstance(S, np.ndarray):
                S = torch.from_numpy(S.astype('float32')).to(torch.get_default_device()).contiguous()
            else:
                raise Exception("S must be a torch tensor or numpy array")

        self.S = S
        self.nflips, self.N = S.shape
        self.device = S.device

        assert 0 <= i <= self.N-1
        self.i = i

        assert 0 <= holdout_fraction <= 1
        self.holdout_fraction = holdout_fraction
        self.holdout_shuffle  = holdout_shuffle

        self.num_chunks = num_chunks

        assert linsolve_eps >= 0
        self.linsolve_eps = linsolve_eps
            
        # Used for copying of object
        self._init_args = ['i', 'holdout_fraction', 'holdout_shuffle', 'num_chunks', 'linsolve_eps']

        
    def split_train_test(self):
        # Split current data set into training part and heldout testing part
        if not hasattr(self, 'trn_tst_split_'): # training and testign splits are cached
            if self.holdout_shuffle:
                perm = np.random.permutation(self.nflips)
                S = self.S[perm]
            else:
                S = self.S

            trn_nflips = self.nflips - int(self.nflips*self.holdout_fraction)

            kw_args = {k : getattr(self,k) for k in self._init_args}
            trn = EPEstimators(S[:trn_nflips,:], **kw_args)
            tst = EPEstimators(S[trn_nflips:,:], **kw_args)
            self.trn_tst_split_ = trn, tst

        return self.trn_tst_split_


    # ====================================================================================
    # Methods to compute observable statistics under forward distribution
    # ====================================================================================
    def g_mean(self):
        # Compute means of g observables
        if not hasattr(self, 'g_mean_'): # Cache values for speed
            i = self.i
            g = (-2 * self.S[:, i]) @ self.S / self.nflips 
            self.g_mean_ = remove_i(g, i)
        return self.g_mean_

    def g_secondmoments(self):
        # Compute matrix of second moments of g observables
        if not hasattr(self, 'g_secondmoments_'): # Cache values for speed
            if self.num_chunks is None:
                K = (4 * self.S.T) @ self.S
            else:
                # Chunked computation
                K = torch.zeros((self.N, self.N), device=self.device)
                chunk_size = (self.nflips + self.num_chunks - 1) // self.num_chunks  # Ceiling division

                for r in range(self.num_chunks):
                    start = r * chunk_size
                    end = min((r + 1) * chunk_size, self.nflips)
                    S_chunk = self.S[start:end]
                    
                    K += (4 * S_chunk.T) @ S_chunk

            K /= self.nflips

            self.g_secondmoments_ = remove_i_rowcol(K, self.i)
        return self.g_secondmoments_

    def g_covariance(self): 
        # Compute covariance matrix of g observables
        if not hasattr(self, 'g_covariance_'): # Cache for speed
            gmean = self.g_mean()
            self.g_covariance_ = self.g_secondmoments() - torch.outer(gmean, gmean)
        return self.g_covariance_

    # ====================================================================================
    # Methods to compute observable statistics under tilted reverse distribution
    # ====================================================================================

    def _get_tilted_statistics(self, theta, return_mean=False, return_covariance=False, return_objective=False):
        # This internal method computes tilted statistics reverse distribution titled by theta. This may include
        # the objective ( θᵀ<g> - ln(<exp(θᵀg)>_{~p}) ), the tilted mean, and the tilted covariance

        assert return_mean or return_covariance or return_objective

        vals       = {}
        i          = self.i

        theta_pad  = add_i(theta, i)

        if not hasattr(self, '_2SiS'): # cache for faster computations
            self._2SiS = torch.einsum('j,ji->ji', -2 * self.S[:,i], self.S).contiguous()

        th_g       = self._2SiS @ theta_pad
        th_g_min   = torch.min(th_g)    # substract max of -th_g for numerical stability
        Y          = torch.exp(-th_g+th_g_min)
        norm_const = torch.mean(Y)
        S1_S       = -(-2 * self.S[:, i]) * Y

        if return_objective:
            # trueZ = Z*torch.exp(-th_g_min)
            log_Z             = torch.log(norm_const) - th_g_min
            vals['objective'] = float( theta @ self.g_mean() - log_Z )

        if return_mean or return_covariance:
            mean       = S1_S @ self.S / (self.nflips * norm_const)
            mean_noi   = remove_i(mean, i)

            if return_mean:
                vals['tilted_mean'] = mean_noi
            
            if return_covariance:
                if self.num_chunks is None:
                    K = (4 * Y * self.S.T) @ self.S
                else:
                    # Chunked computation
                    K = torch.zeros((self.N, self.N), device=self.device)
                    chunk_size = (self.nflips + self.num_chunks - 1) // self.num_chunks  # Ceiling division

                    for r in range(self.num_chunks):
                        start = r * chunk_size
                        end = min((r + 1) * chunk_size, self.nflips)
                        S_chunk = self.S[start:end]
                        
                        th_g_chunk = (-2 * S_chunk[:, i]) * (S_chunk @ theta_pad)
                        K += (4 * torch.exp(-th_g_chunk+th_g_min) * S_chunk.T) @ S_chunk

                K /= (self.nflips * norm_const)

                vals['tilted_covariance'] = remove_i_rowcol(K,i) - torch.outer(mean_noi,mean_noi)

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
        elif objective >= np.log(self.nflips):
            # EP estimate should not be larger than log(self.nflips), because it is not possible
            # to estimate KL divergence larger than log(m) from m samples
            objective = np.log(self.nflips)
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
        #     theta     = solve_linear_psd(A, 2*self.g_mean())
        #     tst_objective = None

        A         = self.g_secondmoments()
        A        += self.linsolve_eps*eye_like(A)
        theta     = solve_linear_psd(A, 2*self.g_mean())
        sigma     = float(theta @ self.g_mean())
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

        i      = self.i 
        if holdout:
            trn, tst = self.split_train_test()
            f_cur_trn = f_cur_tst = f_new_trn = f_new_tst = 0.0
        else:
            trn = self
            f_cur_trn = f_new_trn = 0.0
        
        theta = torch.zeros(self.N-1, device=self.device)
        I     = torch.eye(self.N-1, device=self.device)

        for _ in range(max_iter):
            # Find Newton step direction. We first get gradient and Hessian
            stats = trn._get_tilted_statistics(theta, return_mean=True, return_covariance=True)
            g_theta = stats['tilted_mean']
            H_theta = stats['tilted_covariance']

            if is_infnan(H_theta.sum()): 
                # Error occured, usually it means theta is too big
                if verbose: print('invalid Hessian in get_EP_Newton_steps')
                break

            grad = trn.g_mean() - g_theta
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
            elif f_new_trn > np.log(trn.nflips):  
                # One cannot estimate KL divergence larger than log(nflips) from sample of size nflips.
                # This is a signature of undersampling; when it happens, we clip our estimate of the 
                # objective and exit
                f_new_trn = np.log(trn.nflips)
                last_round = True

            if holdout:                # Do the same checks but now on the heldout test data
                f_new_tst = tst.get_objective(new_theta) 
                if is_infnan(f_new_tst) or f_new_tst <= f_cur_tst:
                    break
                elif np.abs(f_new_tst - f_cur_tst) <= tol:
                    last_round = True
                elif f_new_tst > np.log(tst.nflips):
                    f_new_tst = np.log(tst.nflips)
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
        
        i           = self.i 
        S           = trn.S.T.contiguous()   # Transposing seems to lead to faster calculations

        if theta_init is not None:
            new_theta = theta_init
        else:
            new_theta = torch.zeros(self.N-1, device=self.device)

        m = torch.zeros_like(new_theta)
        v = torch.zeros_like(new_theta)

        for t in range(max_iter):
            tilted_stats = trn._get_tilted_statistics(new_theta, return_objective=True, return_mean=True)
            f_new_trn    = tilted_stats['objective']

            grad         = trn.g_mean() - tilted_stats['tilted_mean']   # Gradient


            # Different conditions that will stop optimization. See get_EP_Newton above
            # for a description of different branches
            last_round = False # flag that indicates whether to break after updating values
            if is_infnan(f_new_trn) or f_new_trn <= f_cur_trn:
                break
            elif f_new_trn > np.log(trn.nflips):
                f_new_trn = np.log(trn.nflips)
                last_round = True
            elif np.abs(f_new_trn - f_cur_trn) <= tol: 
                last_round = True

            if holdout:
                f_new_tst = tst.get_objective(new_theta) 
                if is_infnan(f_new_tst) or f_new_tst <= f_cur_tst:
                    break
                elif f_new_tst > np.log(tst.nflips):
                    f_new_tst = np.log(tst.nflips)
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

