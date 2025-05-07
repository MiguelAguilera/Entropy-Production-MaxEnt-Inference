# Some unit testing, can be run using 
# > pytest tests.py

import os, sys
import numpy as np
from tqdm import tqdm

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable torch fallback for MPS backend
import torch

import spin_model
import ep_estimators
import utils

utils.set_default_torch_device()

def get_simulation_results():
    N    = 10   # system size
    k    = 6    # avg number of neighbors in sparse coupling matrix
    beta = .5   # inverse temperature


    np.random.seed(42) # Set seed for reproducibility

    J    = spin_model.get_couplings_random(N=N, k=k)
    S, F = spin_model.run_simulation(beta=beta, J=J, samples_per_spin=1000, progressbar=False)

    return beta, J, S, F

def test_coupling_matrix_generator():
    N    = 50
    J    = spin_model.get_couplings_random(N=N, k=5)
    J    = spin_model.get_couplings_random(N=N)
    J    = spin_model.get_couplings_patterns(N=N, L=10)

def test_simulation():
    get_simulation_results()


def run_inference(beta, J, S, F, num_chunks=None, do_rev=False):
    num_samples_per_spin, N = S.shape
    total_flips = N * num_samples_per_spin  # Total spin-flip attempts

    # Running sums to keep track of EP estimates
    sigma_emp = sigma_g = sigma_gH = sigma_g2 = sigma_g2H = sigma_N1 = sigma_MTUR = 0.0

    # Because system is multipartite, we can separately estimate EP for each spin
    for i in tqdm(range(N)):
        p_i            =  F[:,i].sum()/total_flips               # frequency of spin i flips

        # Select states in which spin i flipped and use it create object for EP estimation 
        g_samples  = spin_model.get_g_observables(S, F, i)
        data = ep_estimators.Dataset(g_samples=g_samples, rev_g_samples=-g_samples if do_rev else None)
        obj = ep_estimators.EPEstimators(data=data)

        # Empirical estimate 
        J_without_i = np.hstack([J[i,:i], J[i,i+1:]])
        spin_emp = beta * float(utils.numpy_to_torch(J_without_i) @ utils.numpy_to_torch(data.g_mean))
        
        spin_N1   = obj.get_EP_Newton(max_iter=1, num_chunks=num_chunks).objective # 1-step of Newton
        
        # Full optimization with trust-region Newton method and no holdout 
        spin_full = obj.get_EP_Newton(trust_radius=1/4, holdout=False, num_chunks=num_chunks).objective

        # Full optimization with trust-region Newton method and holdout
        np.random.seed(1) # Set seed for holdout reproducibility 
        spin_fullH = obj.get_EP_Newton(trust_radius=1/4, holdout=True, num_chunks=num_chunks).objective

        # Full optimization with gradient ascent method , no holdout
        spin_grad = obj.get_EP_GradientAscent(holdout=False).objective

        # Full optimization with gradient ascent method 
        np.random.seed(1) # Set seed for holdout reproducibility 
        spin_gradH = obj.get_EP_GradientAscent(holdout=True).objective

        # Multidimensional TUR
        spin_MTUR = obj.get_EP_MTUR().objective
        

        sigma_emp  += p_i * spin_emp
        sigma_g    += p_i * spin_full
        sigma_gH   += p_i * spin_fullH
        sigma_N1   += p_i * spin_N1
        sigma_MTUR += p_i * spin_MTUR
        sigma_g2   += p_i * spin_grad
        sigma_g2H  += p_i * spin_gradH

        utils.empty_torch_cache()
        return (sigma_emp, sigma_g, sigma_gH, sigma_g2, sigma_g2H, sigma_N1, sigma_MTUR)
    
def test_inference():
    beta, J, S, F = get_simulation_results()
    run_inference(beta, J, S, F)


def test_inference_chunks():
    beta, J, S, F = get_simulation_results()
    
    results1 = np.array(run_inference(beta, J, S, F))
    for num_chunks in [-1, 10]:
        for do_rev in [True, False]:
            results2 = np.array( run_inference(beta, J, S, F, num_chunks=num_chunks, do_rev=do_rev) )
            print(num_chunks, do_rev, results1 - results2)
            assert(np.allclose(results1 , results2 ))



def test_objective():
    nobservables = 9
    samp = np.random.randn(100,nobservables)
    data = ep_estimators.Dataset(g_samples=samp, rev_g_samples=-samp)
    assert(np.isclose( data.get_objective(torch.zeros(nobservables)),0))



def test_consistency():


    N    = 10     # system size
    k    = 6      # avg number of neighbors in sparse coupling matrix
    beta = 3.25   # inverse temperature


    np.random.seed(42) # Set seed for reproducibility

    J    = spin_model.get_couplings_random(N=N, k=k)
    S, F = spin_model.run_simulation(beta=beta, J=J, samples_per_spin=1000)

    sigma_emp = spin_model.get_empirical_EP(beta, J, S, F)

    X0, X1 = spin_model.convert_to_nonmultipartite(S, F)

    N = J.shape[0]
    g_samples = np.vstack([ X1[:,i]*X0[:,j] - X0[:,i]*X1[:,j] 
                            for i in range(N) for j in range(i+1, N) ]).T
    data1         = ep_estimators.Dataset(g_samples=g_samples, rev_g_samples=-g_samples)
    estimator1    = ep_estimators.EPEstimators(data1)
    sigma_g2_obs  = estimator1.get_EP_GradientAscent(holdout=True).objective

    data2          = ep_estimators.RawDataset(X0, X1)
    estimator2     = ep_estimators.EPEstimators(data2)
    sigma_g2_state = estimator2.get_EP_GradientAscent(holdout=True).objective

    data3         = ep_estimators.Dataset(g_samples=g_samples)
    estimator3    = ep_estimators.EPEstimators(data3)
    sigma_g3_obs  = estimator3.get_EP_GradientAscent(holdout=True).objective


    theta = np.random.rand(data1.nobservables)
    assert(torch.norm(data1.g_mean - data2.g_mean)<1e-5)
    assert(np.abs(data1.get_objective(theta)-data2.get_objective(theta))<1e-5)
    assert(torch.norm(data1.get_tilted_statistics(theta=theta, return_mean=True)['tilted_mean']-
                      data2.get_tilted_statistics(theta=theta, return_mean=True)['tilted_mean'])<1e-5)

    assert(torch.norm(data1.g_mean - data3.g_mean)<1e-5)
    assert(np.abs(data1.get_objective(theta)-data3.get_objective(theta))<1e-5)
    assert(torch.norm(data1.get_tilted_statistics(theta=theta, return_mean=True)['tilted_mean']-
                      data3.get_tilted_statistics(theta=theta, return_mean=True)['tilted_mean'])<1e-5)


