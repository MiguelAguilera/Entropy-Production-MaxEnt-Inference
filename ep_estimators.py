# ============================================================
# Entropy production (EP) estimation methods
# ============================================================

import numpy as np

import optimizers
import linear_solvers
from utils import numpy_to_torch, torch_to_numpy



# ============================================================
# Entropy production (EP) estimation methods
# ============================================================

def get_EP_Estimate(data, validation=None, test=None, verbose=0, max_iter=None,
                    optimizer='GradientDescentBB', optimizer_kwargs=None, **kwargs):
    # This is the main function used to estimate the EP from data
    #
    # Arguments:
    #   data        : DatasetBase object containing samples of observables or state transitions
    #   validation  : DatasetBase object containing samples of observables or state transitions
    #               : that is used for early stopping. if None, we do not do early-stopping
    #   test        : DatasetBase object containing samples of observables or state transitions
    #               : that is used for evaluating the objective. 
    #               : If None, we evaluate the objective on the validation or training set
    #   verbose     : verbosity level (0 = no output, 1 = some output, 2 = all output)
    #   max_iter    : maximum number of iterations for the optimizer
    #   optimizer   : optimizer to use for optimization. This can be either a string specifying a
    #               : an instance of an Optimizer object, or a string specifying built-in optimizer 
    #               : class from optimizers.py. Currently we suppose the following optimizers:
    #               :   'GradientDescent'   - using first order gradient information and fixed learning rate
    #               :   'GradientDescentBB' - using first order gradient information and Barzilai-Borwein step sizes
    #               :   'Adam'              - Adam optimizer
    #               :   'NewtonMethod'      - Newton-Raphson method using second order (Hessian) information 
    #               :   'NewtonMethodTrustRegion' - Newton-Raphson method with trust region constraints
    #               :   'TRON'              - Trust Region Newton method with adaptive trust region
    #   optimizer_kwargs : additional arguments to pass to the optimizer constructor (if optimizer is a string)
    #               :    see optimizers.py for details. Here is an example
    #   kwargs      : any additional arguments to pass to the optimize function
    #
    #  > v, x = get_EP_Estimate(data, optimizer='Adam', optimizer_kwargs={'lr': 0.01,'beta2': 0.9})
    # Returns:
    #   (ep_estimate, theta): the estimated EP and the optimal parameter

    x0 = np.zeros(data.nobservables)
    if data.nsamples == 0 or \
        (validation is not None and validation.nsamples == 0) or \
        (test       is not None and test.nsamples       == 0):
        # There is not enough data to estimate the objective
        return 0, x0
    
    if validation is not None and validation.nsamples > 0:
        val_nsamples = np.log(validation.nsamples) 
    else:
        val_nsamples = None

    o = optimizers.optimize(x0=x0, 
            objective=data, validation=validation, max_iter=max_iter, minimize=False, verbose=verbose, 
            optimizer=optimizer, optimizer_kwargs=optimizer_kwargs,
            max_trn_objective=np.log(data.nsamples), max_val_objective=val_nsamples, **kwargs)
    
    if test is not None and test.nsamples > 0: # Evaluate the objective on the test set
        # cap the objective at the maximum value
        ret_objective = min(test.get_objective(o.x), np.log(test.nsamples))
    elif validation is not None:
        ret_objective = o.val_objective
    else:
        ret_objective = o.objective
    return ret_objective, torch_to_numpy(o.x) 


def get_EP_Newton1Step(data, validation=None, test=None, verbose=0, **optimize_args):
    # Estimate EP using 1 step of Newton method
    optimizer = optimizers.NewtonMethod(verbose=verbose)
    return get_EP_Estimate(data, validation=validation, test=test, optimizer=optimizer, max_iter=1,
                           verbose=verbose, report_every=1, **optimize_args)


def get_EP_MTUR(data, linsolve_eps=1e-4):
    # Estimate EP using the multidimensional TUR method
    # The MTUR is defined as ln(1 + 2 <g>_p^T K^-1 <g>_p)
    # where μ = (p + ~p)/2 is the mixture of the forward and reverse distributions
    # and K^-1 is covariance matrix of g under (p + ~p)/2.
    # 
    # Optional arguments
    #   linsolve_eps (float)     : regularization parameter for covariance matrices, used to improve
    #                               numerical stability of linear solvers

    if data.nsamples == 0:  # There is not enough data to estimate the objective
        return 0, numpy_to_torch(np.zeros(data.nobservables))

    # Compute mean and covariance
    mean_g = data.g_mean
    cov = data.get_covariance()

    # Regularized covariance inverse solve
    reg_cov = cov + linsolve_eps * linear_solvers.eye_like(cov)
    x = linear_solvers.solve_linear_psd(reg_cov, mean_g)

    # Objective value based on quadratic form
    objective = np.log(1 + 2 * float(mean_g @ x))

    return objective, torch_to_numpy(x)

