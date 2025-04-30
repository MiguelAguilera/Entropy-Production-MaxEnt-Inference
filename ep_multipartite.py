# ======================================================
# Contains method to estimate EP in multipartite systems
# ======================================================

# TODO: Explain that this is for estimating EP in multipartite
# system, also explain structure of S passed into our object,
# and structure of observables g that we assume

import os, time
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]='1'

import numpy as np
import scipy
import torch


from utils import * 



class EPEstimators(object):
    def __init__(self, S, i, num_chunks=None, linsolve_args={}):
        # TODO: Explain input format of S
        #
        # Parameters
        # ----------
        # num_chunks : int
        #   chunk covariance computations to reduce memory requirements
        # linsolve_eps : float 
        #   regularize covariance matrices for numerical stability of linear solver
        # linsolve_method : str
        #   method used to solve linear system (None for default)

        self.S = S
        self.i = i
        self.device = S.device
        self.num_chunks = num_chunks
        self.nflips, self.N = S.shape

        self.linsolve_args = linsolve_args.copy()
        if 'eps' not in linsolve_args:
            self.linsolve_args['eps'] = 1e-4
            
        # Store args for easy copying later
        self._init_args = dict(
            i=self.i,
            num_chunks=num_chunks,
            linsolve_args=self.linsolve_args,
        )
        
    def spawn(self, S_new):
        return EPEstimators(S_new, **self._init_args)

    def g_mean(self):
        # Compute  means of g observables
        if not hasattr(self, 'g_mean_'): # Cache value
            i = self.i
            g = (-2 * self.S[:, i]) @ self.S / self.nflips 
            self.g_mean_ = remove_i(g, i)
        return self.g_mean_

    def g_secondmoments(self):
        # Compute matrix of second moments of g observables
        if not hasattr(self, 'g_secondmoments_'): # Cache for speed
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

                    empty_cache()
            K /= self.nflips

            self.g_secondmoments_ = remove_i_rowcol(K, self.i)
        return self.g_secondmoments_

    def g_covariance(self): 
        # Compute covariance matrix of g observables
        if not hasattr(self, 'g_covariance_'): # Cache for speed
            gmean = self.g_mean()
            self.g_covariance_ = self.g_secondmoments() - torch.outer(gmean, gmean)
        return self.g_covariance_

    def g_mean_theta(self, theta):
        # Compute expectation of g under reverse distribution titled by theta
        i = self.i
        theta_padded = add_i(theta, i)
        th_g     = (-2 * self.S[:, i]) * (self.S @ theta_padded)
        th_g_min = torch.min(th_g)    # substract max of -th_g for numerical stability
        Y        = torch.exp(-th_g+th_g_min)
        Z        = torch.sum(Y) / self.nflips
        S1_S     = -(-2 * self.S[:, i]) * Y

        mean = S1_S @ self.S / (self.nflips * Z)
        return remove_i(mean, i)


    def g_mean_and_covariance_theta(self, theta):
        # Compute expectation and covariance of g under reverse distribution titled by theta
        i = self.i
        theta_padded = add_i(theta, i)
        th_g     = (-2 * self.S[:, i]) * (self.S @ theta_padded)
        th_g_min = torch.min(th_g)    # substract max of -th_g for numerical stability
        Y        = torch.exp(-th_g+th_g_min)
        Z        = torch.sum(Y) / self.nflips
        S1_S     = -(-2 * self.S[:, i]) * Y

        mean = S1_S @ self.S / (self.nflips * Z)
    
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
                
                th_g_chunk = (-2 * S_chunk[:, i]) * (S_chunk @ theta_padded)
                K += (4 * torch.exp(-th_g_chunk+th_g_min) * S_chunk.T) @ S_chunk

                empty_cache()

        K /= (self.nflips * Z)

        # trueZ = Z*torch.exp(-th_g_min)
        mean_noi = remove_i(mean, i)
        return mean_noi, remove_i_rowcol(K,i) - torch.outer(mean_noi,mean_noi)


    def log_norm_theta(self, theta):
        # Estimate log normalization constant Z for subproblem i under theta.
        # TODO : maybe remove if its not needed
        i = self.i
        theta_padded = add_i(theta,i)
        th_g = (-2 * self.S[:, i]) * (self.S @ theta_padded)
        th_g_min = torch.min(th_g)    # substract max of -th_g
        return float(torch.log(torch.mean(torch.exp(-th_g+th_g_min))) - th_g_min)

    def get_objective(self, theta):
        # Return objective value for parameters theta
        return float(theta @ self.g_mean()) - self.log_norm_theta(theta)


    def newton_step(self, theta_init, trust_radius=None):
        """
        Perform a Newton-Raphson update to refine the parameter theta.

        Parameters:
        -----------
        theta_init : torch.Tensor
            Current estimate of the parameter vector theta (with zero at index i).
        trust_radius : float or None
            trust_radius to pass into linear solver

        Returns:
        --------
        delta_theta : torch.Tensor
            direction of update
        """

        i = self.i
        g_theta, K_theta = self.g_mean_and_covariance_theta(theta=theta_init)
        
        if is_infnan(K_theta.sum()):
            # Error occured, usually means theta is too big
            return np.nan, theta_init*np.nan

        rhs = self.g_mean() - g_theta

        ls_args = self.linsolve_args.copy()
        if trust_radius is not None:
            ls_args['trust_radius'] = trust_radius

        # Compute Newton step: Δθ = H⁻¹ (g - g_theta)
        return solve_linear_psd(K_theta, rhs, **ls_args)



    # =======================
    # EP estimation methods 
    # =======================

    def get_EP_MTUR(self):
        # Compute entropy production estimate using the MTUR method
        #    method (str) : which method to use to solve linear system
        gmean = self.g_mean()
        theta = solve_linear_psd(self.g_secondmoments(), 2*gmean, **self.linsolve_args)
        return float(theta @ gmean)


    def get_EP_Newton(self):
        # Estimate EP using the 1-step Newton method for spin i.
        if not hasattr(self, 'newton_1step_'):
            # Cache the returned values as they are used by several other methods
            ls_args = self.linsolve_args.copy()
            assert 'eps' in ls_args

            while True:  # regularize until we get a non-zero Z
                if ls_args['eps'] > 1:
                    print('get_EP_Newton cannot regularize enough!')
                    return np.nan, theta
                theta = solve_linear_psd(self.g_covariance(), 2*self.g_mean(), **ls_args)

                # Z = self.norm_theta(theta)
                theta_padded = add_i(theta, self.i)
                th_g = (-2 * self.S[:, self.i]) * (self.S @ theta_padded)
                Z    = torch.mean(torch.exp(-th_g))

                if torch.abs(Z)>1e-30 and not is_infnan(Z):
                    break

                ls_args['eps'] *= 10

            sig_N1 = float(theta @ self.g_mean() - torch.log(Z))
            self.newton_1step_ = (sig_N1, theta)

        return self.newton_1step_



    def get_EP_Newton_steps(self, holdout=False, tol=1e-4, max_iter=1000, newton_step_args={}):
        i      = self.i 
        nflips = self.nflips 

        if holdout:
            nflips = int(self.nflips/2)

            trn = self.spawn(self.S[:nflips,:])
            tst = self.spawn(self.S[nflips:,:])
        else:
            nflips = self.nflips
            trn = self



        # sig_old_trn, theta = trn.get_EP_Newton()
        # sig_old_tst = tst.get_objective(theta)

        # if is_infnan(sig_old_tst) or sig_old_tst <= 0:
        #     return 0.0, torch.zeros(self.N-1) 

        sig_old_trn = sig_old_tst = sig_new_trn = sig_new_tst = 0.0
        
        theta = torch.zeros(self.N - 1, device=self.device)

        max_norm = 1/4

        for _ in range(max_iter):
            
            # **** Find Newton step direction
            delta_theta = trn.newton_step(theta_init=theta)
            delta_theta *= max_norm/max(max_norm, torch.norm(delta_theta))
            new_theta    = theta + delta_theta
                
            sig_new_trn  = self.get_objective(new_theta)

            if is_infnan(sig_new_trn) or sig_new_trn <= sig_old_trn or sig_new_trn > np.log(nflips):
                break

            if holdout:
                sig_new_tst = tst.get_objective(new_theta) 
                if is_infnan(sig_new_tst) or sig_new_tst <= sig_old_tst or sig_new_tst > np.log(nflips):
                    break

            last_round = False
            if np.abs(sig_new_trn - sig_old_trn) <= tol: # *np.abs(sig_old_trn):
                last_round = True

            if holdout and np.abs(sig_new_tst - sig_old_tst) <= tol: # *np.abs(sig_old_tst):
                last_round = True

            sig_old_tst, sig_old_trn, theta = sig_new_tst, sig_new_trn, new_theta

            if last_round:
                break

            # if sig_new_tst <= sig_old_tst:
                    
            #     # delta_theta = new_theta - theta
            #     # def dfunc(x):
            #     #     v = theta + x * delta_theta
            #     #     return float( (tst.g_mean() - tst.g_mean_theta(v))@delta_theta )

            #     # if np.sign(dfunc(0)) == np.sign(dfunc(1)):
            #     #     break

            #     # x_max = scipy.optimize.bisect(dfunc, 0, 1, rtol=1e-3)
            #     # theta = theta + x_max * delta_theta
            #     # sig_old_tst = tst.get_objective(theta)

            #     break



        # return self.get_objective(theta), theta
        return sig_old_tst if holdout else sig_old_trn, theta



    def get_EP_TRON(self, tol=1e-3, max_iter=100, 
                    trust_radius_init=0.5, trust_radius_max=1000.0,
                    eta0=0.0, eta1=0.25, eta2=0.75, tol_val=0,
                    holdout=True, tron=True):

        nflips = int(self.nflips / 2)
        i = self.i

        perm = np.random.permutation(self.S.shape[0])
        S_shuffled = self.S[perm]
        if holdout:
            trn = self.spawn(S_shuffled[:nflips,:])
            tst = self.spawn(S_shuffled[nflips:,:])
        else:
            trn = self.spawn(self.S)
            tst = None  # unused

        f_old_trn, theta = self.get_EP_Newton()

        trust_radius = trust_radius_init

        g = trn.g_mean()
        g_theta, H_theta = trn.g_mean_and_covariance_theta(theta)
        grad = g - g_theta
        grad_norm = grad.abs().max()

        f_old_val = tst.get_objective(theta) if holdout else None

        for iteration in range(max_iter):
            if grad_norm < tol:
                break

            p = steihaug_toint_cg(A=H_theta, b=grad, trust_radius=trust_radius)
            pred_red = grad @ p + 0.5 * p @ (H_theta @ p)

            theta_new = theta + p
            f_new_trn = trn.get_objective(theta_new)
            act_red = f_new_trn - f_old_trn
            rho = act_red / pred_red

            # Accept step
            if rho > eta0 or (not tron):
                if holdout:
                    f_new_val = tst.get_objective(theta_new)
                    if f_new_val + tol_val * trust_radius < f_old_val:
                        break
                theta = theta_new
                f_old_trn = f_new_trn
                if holdout:
                    f_old_val = f_new_val
                g_theta, H_theta = trn.g_mean_and_covariance_theta(theta)
                grad = g - g_theta
                grad_norm = grad.abs().max()

            # Update trust radius
            if tron:
                if rho < eta1:
                    trust_radius *= 0.25
                elif rho > eta2 and p.norm() == trust_radius:
                    trust_radius = min(2.0 * trust_radius, trust_radius_max)
        
        return self.get_objective(theta), theta


    def get_EP_Adam(self, theta_init, holdout=False, max_iter=1000, 
                    beta1=0.9, beta2=0.999, lr=0.01, eps=1e-8, 
                    tol=1e-4, skip_warm_up=False,
                    timeout=60):
        """
        Performs multiple Adam-style updates to refine theta estimation.
        
        Arguments:
            S         : binary spin samples, shape (num_flips,N)
            theta_init: initial theta vector
            g         : empirical expectation vector
            i         : index to remove from theta (current spin)
            num_iters : number of Adam updates
            beta1, beta2: Adam moment decay parameters
            lr        : learning rate
            eps       : epsilon for numerical stability
            tol       : tolerance for early stopping
            skip_warm_up : Adam option
            timeout : maximum second to run
        
        Returns:
            sig_gd    : final entropy production estimate
            theta     : final updated theta
        """

        i      = self.i 
        nflips = self.nflips

        if holdout:
            nflips = int(self.nflips/2)
            trn = self.spawn(self.S[:nflips,:])
            tst = self.spawn(self.S[nflips:,:])
        else:
            trn = self
        
        trn_g_withi = add_i(trn.g_mean(),i)


        S      = trn.S.T.contiguous()   # Transpose for quicker calculations

        theta = add_i(theta_init, i)

        m = torch.zeros_like(theta)
        v = torch.zeros_like(theta)

        stime = time.time()
        twice_S_onlyi = 2*S[i,:]

        X = (-torch.einsum('j,ij->ij', twice_S_onlyi, S)).contiguous() # S_onlyi[:,None] * S_i
        # -2 * S_onlyi * S_i # 
        last_val = -np.inf
        cur_val  =  np.nan

        for t in range(1, max_iter + 1):
            th_g = theta@X
            
            Y = torch.exp(-th_g)
            Z = torch.mean(Y)
            S1_S = twice_S_onlyi * Y 

            g_theta = torch.einsum('r,jr->j', S1_S, S) / (nflips * Z)

            if holdout:
                cur_val = tst.get_objective(remove_i(theta,i)) 
            else:
                cur_val = float(theta @ trn_g_withi - torch.log(Z))

            if is_infnan(cur_val):
                break
            #if cur_val > np.log(nflips):
            #    break
            # Early stopping
            if t>5 and ((time.time()-stime > timeout) or (np.abs((last_val - cur_val)/last_val) < tol)):
                break

            last_val = cur_val

            grad = trn_g_withi - g_theta

            if False:
                # regular gradient descent
                theta += lr * grad
                
            else:
                # Adam moment updates
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad**2)
                if skip_warm_up:
                    m_hat = m
                    v_hat = v
                else:
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)

                # Compute parameter update
                delta_theta = lr * m_hat / (v_hat.sqrt() + eps)

                theta += delta_theta

            theta[i]=0.0

        return cur_val, remove_i(theta,i)



