# Some unit testing

import os, sys
import numpy as np
from tqdm import tqdm

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable torch fallback for MPS backend
import torch

sys.path.insert(0, '..')
import spin_model
import ep_multipartite
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


def run_inference(beta, J, S, F, num_chunks=None):
    num_samples_per_spin, N = S.shape
    total_flips = N * num_samples_per_spin  # Total spin-flip attempts

    # Running sums to keep track of EP estimates
    sigma_emp = sigma_g = sigma_g2 = sigma_N1 = sigma_MTUR = 0.0

    with torch.no_grad():

        # Because system is multipartite, we can separately estimate EP for each spin
        for i in tqdm(range(N)):
            p_i            =  F[:,i].sum()/total_flips               # frequency of spin i flips

            # Select states in which spin i flipped and use it create object for EP estimation 
            S_i = S[F[:,i],:]
            obj = ep_multipartite.EPEstimators(S_i, i, num_chunks=num_chunks)

            # Empirical estimate 
            g_expectations = utils.add_i( obj.g_mean(), i ).cpu().numpy()
            spin_emp = beta * J[i,:] @ g_expectations

            spin_MTUR = obj.get_EP_MTUR().objective             # Multidimensional TUR
            spin_N1   = obj.get_EP_Newton(max_iter=1).objective # 1-step of Newton
            
            # Full optimization with trust-region Newton method and holdout 
            spin_full = obj.get_EP_Newton(trust_radius=1/4, holdout=True).objective

            # Full optimization with gradient ascent method 
            spin_grad = obj.get_EP_GradientAscent(holdout=True).objective

            sigma_emp  += p_i * spin_emp
            sigma_g    += p_i * spin_full
            sigma_N1   += p_i * spin_N1
            sigma_MTUR += p_i * spin_MTUR
            sigma_g2   += p_i * spin_grad

            utils.empty_torch_cache()

        return sigma_emp, sigma_g, sigma_g2, sigma_N1, sigma_MTUR
    
def test_inference():
    beta, J, S, F = get_simulation_results()
    run_inference(beta, J, S, F)


def test_inference_chunks():
    beta, J, S, F = get_simulation_results()
    
    sigma_empA, sigma_gA, sigma_g2A, sigma_N1A, sigma_MTURA = run_inference(beta, J, S, F)
    sigma_empB, sigma_gB, sigma_g2B, sigma_N1B, sigma_MTURB = run_inference(beta, J, S, F, num_chunks=10)

    assert(np.isclose(sigma_empA , sigma_empB ))
    assert(np.isclose(sigma_gA   , sigma_gB   ))
    assert(np.isclose(sigma_g2A  , sigma_g2B  ))
    assert(np.isclose(sigma_N1A  , sigma_N1B  ))
    assert(np.isclose(sigma_MTURA, sigma_MTURB))
