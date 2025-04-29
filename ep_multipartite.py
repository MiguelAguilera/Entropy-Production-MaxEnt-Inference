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
    def __init__(self, S, i, num_chunks=None, linsolve_eps=1e-4, linsolve_method=None):
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

        self.linsolve_eps = 1e-4  # regularize covariance matrices to solve linear system
        self.linsolve_method = linsolve_method


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


    def newton_step(self, theta_init, delta=None, th=None, logZpad=1e-5,do_linesearch=False):
        """
        Perform one iteration of a constrained Newton-Raphson update to refine the parameter theta.

        Parameters:
        -----------
        theta_init : torch.Tensor
            Current estimate of the parameter vector theta (with zero at index i).
        th : float
            Threshold ?
        delta : float or None, optional
            If provided, sets a maximum relative norm for the update step 
            (default: 1.0, meaning ||Δθ|| ≤ delta * ||θ||).

        Returns:
        --------
        sigma : float
            Updated estimate of the objective value
        theta : torch.Tensor
            new theta value
        """

        i = self.i
        g_theta, K_theta = self.g_mean_and_covariance_theta(theta=theta_init)
        
        if torch.isinf(K_theta).any():
            # Error occured, usually means theta is too big
            return np.nan, theta_init*np.nan

        # Compute Newton step: Δθ = H⁻¹ (g - g_theta)
        rhs = self.g_mean() - g_theta
        delta_theta = solve_linear_psd(K_theta, rhs, eps=self.linsolve_eps, method=self.linsolve_method)

        step_size = 1

        if do_linesearch:
            def g(alpha):
                return self.objective(theta_init + alpha * delta_theta)
    
            def dg(alpha):
                df = self.g_mean() - self.g_mean_theta(theta=theta_init + alpha * delta_theta)
                return -df @ delta_theta

            def get_step_size(g, dg):
        
                a, b = 0.0, 1.0                         # Initialize the search interval [0,1]
                 
                initial_derivative = dg(0)              # Initial check for descent direction
                if initial_derivative >= 0:
                    return 0.0  # No step taken
                
                final_derivative = dg(1.0)              # Check if full step is acceptable
                if final_derivative >= 0: # and final_derivative <= tol:
                    return 1.0 
                
                # Apply bisection with Newton's method
                max_iter_ls = 4
                for i in range(max_iter_ls):
                    # Use Newton's method to estimate alpha within the interval
                    if dg(a) != dg(b):  # Avoid division by zero
                        alpha = a - dg(a) * (b - a) / (dg(b) - dg(a))
                    else:
                        alpha = (a + b) / 2.0
                    
                    # If Newton's step falls outside [0,1], use bisection
                    if alpha <= a or alpha >= b:
                        alpha = (a + b) / 2.0
                    
                    # Compute derivative at the new point
                    derivative = dg(alpha)
                    
                    # Check for convergence
                    if abs(derivative) < 1e-3: # tol:
                        return alpha
                    
                    # Update the interval
                    if derivative > 0:
                        b = alpha
                    else:
                        a = alpha
                
                # Return the midpoint of the final interval if max iterations reached
                return (a + b) / 2.0
            step_size = get_step_size(g,dg)
            #print("ls:", step_size)

        elif delta is not None:
            # Constrain the step to be within a trust region
            max_step = delta * torch.norm(theta_init)
            step_norm = torch.norm(delta_theta)
            if step_norm > max_step:
                delta_theta = delta_theta * (max_step / step_norm)
            step_size = delta
        elif th is not None:
            alpha = 1
            d1 = float(delta_theta @ g_theta)
            d2 = float(delta_theta @ (self.g_mean()-g_theta)) / 2
            
            logZ = self.log_norm_theta(theta_init)
            dlogZ = self.log_norm_theta(theta_init + alpha*delta_theta) - logZ
            dlogZ_approx = alpha * d1 + alpha**2 * d2

            while np.abs(dlogZ_approx-dlogZ)>th*np.abs(dlogZ_approx + logZpad):
                alpha *= 0.95
                dlogZ = self.log_norm_theta(theta_init + alpha*delta_theta) - logZ
                dlogZ_approx = alpha * d1 + alpha**2 * d2
            step_size = alpha
        
        theta = theta_init + step_size*delta_theta

        return self.get_objective(theta), theta


    # =======================
    # EP estimation methods 
    # =======================

    def get_EP_MTUR(self):
        # Compute entropy production estimate using the MTUR method
        #    method (str) : which method to use to solve linear system
        gmean = self.g_mean()
        theta = solve_linear_psd(self.g_secondmoments(), 2*gmean, 
                                 eps=self.linsolve_eps, method=self.linsolve_method)
        return float(theta @ gmean)


    def get_EP_Newton(self):
        # Estimate EP using the 1-step Newton method for spin i.
        if not hasattr(self, 'newton_1step_'):
            # Cache the returned values as they are used by several other methods
            eps = self.linsolve_eps

            while True:  # regularize until we get a non-zero Z
                if eps > 1:
                    print('get_EP_Newton cannot regularize enough!')
                    return np.nan, theta
                theta = solve_linear_psd(self.g_covariance(), 2*self.g_mean(), 
                                         eps=eps, method=self.linsolve_method)

                # Z = self.norm_theta(theta)
                theta_padded = add_i(theta, self.i)
                th_g = (-2 * self.S[:, self.i]) * (self.S @ theta_padded)
                Z    = torch.mean(torch.exp(-th_g))

                if torch.abs(Z)>1e-30 and not is_infnan(Z):
                    break

                eps *= 10

            sig_N1 = float(theta @ self.g_mean() - torch.log(Z))
            self.newton_1step_ = (sig_N1, theta)

        return self.newton_1step_



    def get_EP_Newton_steps(self, tol=1e-3, max_iter=20, newton_step_args={}):
        # Estimate EP using multiple steps of Newton's method
        sig_new, theta = self.get_EP_Newton()
        for _ in range(max_iter):
            sig_old   = sig_new
            theta_old = theta
            sig_new   = np.nan
            sig_new, theta = self.newton_step(theta_init=theta, **newton_step_args)

            dsig = sig_new - sig_old
            if np.abs(dsig) <= tol*np.abs(sig_old):  # relative change test
                break

            if sig_new > np.log(self.nflips):
                #print(f'Break at iteration {count}: log(nflips)={np.log(nflips):.4e}, sig_new={sig_new:.4e}')
                break 
            if sig_new < sig_old or is_infnan(sig_new):
                # print(f'Break at iteration {count}: sig_old={sig_old:.4e}, sig_new={sig_new:.4e}')
                return sig_old, theta
            
        return sig_new, theta


    def get_EP_Newton_steps_holdout(self, tol=1e-3, max_iter=1000, newton_step_args={}):
        nflips = int(self.nflips/2)
        i      = self.i 

        trn = EPEstimators(self.S[:nflips,:], i)
        tst = EPEstimators(self.S[nflips:,:], i)

        if False:
            sig_old_trn, theta = trn.get_EP_Newton()
            #sig_new, theta  = newton_step(S, theta_init, g, i, num_chunks=num_chunks)
            sig_old_tst = tst.get_objective(theta)

            if is_infnan(sig_old_tst):
                return sig_old_trn, theta 

        else:
            sig_old_trn, sig_old_tst = 0.0, 0.0
            theta = torch.zeros(self.N-1)
            #print('here',tst.get_objective(theta))
            #sig_new_trn, new_theta = trn.newton_step(theta_init=theta)
            #print(sig_new_trn, new_theta)
            #sig_new_trn, new_theta = trn.newton_step(theta_init=theta, **newton_step_args)
            #print(sig_new_trn, new_theta)
            #asdf


        for _ in range(max_iter):
            sig_new_trn, new_theta = trn.newton_step(theta_init=theta, **newton_step_args)
            if is_infnan(sig_new_trn) or sig_new_trn <= sig_old_trn or sig_new_trn > np.log(nflips):
                break

            sig_new_tst = tst.get_objective(new_theta) 
            if is_infnan(sig_new_tst) or sig_new_tst > np.log(nflips):
                break

            if sig_new_tst <= sig_old_tst:
                
                # delta_theta = new_theta - theta
                # def dfunc(x):
                #     v = theta + x * delta_theta
                #     return float( (tst.g_mean() - tst.g_mean_theta(v))@delta_theta )

                # if np.sign(dfunc(0)) == np.sign(dfunc(1)):
                #     break

                # x_max = scipy.optimize.bisect(dfunc, 0, 1, rtol=1e-3)
                # theta = theta + x_max * delta_theta
                # sig_old_tst = tst.get_objective(theta)
                
                break

            if np.abs(sig_new_trn - sig_old_trn) <= tol*np.abs(sig_old_trn) or \
               np.abs(sig_new_tst - sig_old_tst) <= tol*np.abs(sig_old_tst):
                #print('here?')
                break

            sig_old_tst, sig_old_trn, theta = sig_new_tst, sig_new_trn, new_theta

        return sig_old_tst, theta



    def get_EP_TRON(self,  
            max_iter=100, tol=1e-3, 
            trust_radius_init=0.5, trust_radius_max=1.0,
            eta0=0.0, eta1=0.25, eta2=0.75):

        def steihaug_toint_cg(grad, H, trust_radius, tol=1e-10, max_iters=250):
            x = torch.zeros_like(grad)
            r = grad.clone()
            d = r

            for i in range(max_iters):
                Hd = H @ d
                dHd = d @ Hd

                if dHd <= 0:
                    # Negative curvature, move to boundary
                    tau = find_tau(x, d, trust_radius)
                    return x + tau * d

                alpha = (r @ r) / dHd
                x_new = x + alpha * d

                if x_new.norm() >= trust_radius:
                    tau = find_tau(x, d, trust_radius)
                    return x + tau * d

                r_new = r + alpha * Hd

                if r_new.norm() < tol:
                    return x_new

                beta = (r_new @ r_new) / (r @ r)
                d = -r_new + beta * d
                x = x_new
                r = r_new

            return x

        def find_tau(x, d, trust_radius):
            # Solve ||x + tau*d|| = trust_radius for tau
            a = d @ d
            b = 2 * (x @ d)
            c = (x @ x) - trust_radius**2
            sqrt_discriminant = torch.sqrt(b**2 - 4*a*c)
            tau = (-b + sqrt_discriminant) / (2*a)
            return tau


        f_old, theta = self.get_EP_Newton()
        trust_radius = trust_radius_init
        
        for iteration in range(max_iter):
            g_theta, K_theta = self.g_mean_and_covariance_theta(theta=theta)
            grad = self.g_mean() - g_theta
            grad_norm = grad.norm()

            if grad_norm < tol:
                break

            # Solve the trust region subproblem approximately
            # using conjugate gradient with truncation
            p = steihaug_toint_cg(grad, K_theta, trust_radius)

            # Predicted reduction
            pred_red = (grad @ p + 0.5 * p @ (K_theta @ p))

            # Actual reduction
            theta_new = theta + p
            f_new     = self.get_objective(theta_new)
            act_red   = f_new - f_old

            # Ratio of actual to predicted reduction
            rho = act_red / pred_red

            if rho < eta1:
                trust_radius *= 0.25
            elif rho > eta2 and p.norm() == trust_radius:
                trust_radius = min(2.0 * trust_radius, trust_radius_max)

            if rho > eta0:
                theta = theta_new
                f_old = f_new

            #        print(f"Iter {iteration}: f = {f_old:.6f}, ||grad|| = {grad_norm:.6e}, trust_radius = {trust_radius:.4f}")

        return f_old, theta        


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
            trn = EPEstimators(self.S[:nflips,:], i)
            tst = EPEstimators(self.S[nflips:,:], i)
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



