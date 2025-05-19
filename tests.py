# Some unit testing, can be run using 
# > pytest tests.py

import os, sys
import numpy as np
from tqdm import tqdm

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable torch fallback for MPS backend
import utils
utils.set_default_torch_device()

import spin_model
import ep_estimators
import observables

OPTIMIZE_VERBOSE=2

_cached_simulation = None
def get_simulation_results():
    global _cached_simulation
    if _cached_simulation is None:
        N    = 10   # system size
        k    = 6    # avg number of neighbors in sparse coupling matrix
        beta = 3    # inverse temperature

        np.random.seed(42) # Set seed for reproducibility

        J    = spin_model.get_couplings_random(N=N, k=k)
        S, F = spin_model.run_simulation(beta=beta, J=J, samples_per_spin=10000, progressbar=False, seed=42)

        _cached_simulation = beta, J, S, F
    return _cached_simulation

def test_coupling_matrix_generator():
    N    = 50
    J    = spin_model.get_couplings_random(N=N, k=5)
    J    = spin_model.get_couplings_random(N=N)
    J    = spin_model.get_couplings_patterns(N=N, L=10)


def test_simulation():
    get_simulation_results()

def test_EP():
    beta, J, S, F = get_simulation_results()
    spin_model.get_empirical_EP(beta, J, S, F)

def test_NewtonVsDefault():
    beta, J, S, F = get_simulation_results()
    g_samples     = observables.get_g_observables(S, F, 0)
    data          = observables.Dataset(g_samples=g_samples)
    v0, _ = ep_estimators.get_EP_Estimate(data=data, optimizer=optimizers.NewtonMethodTrustRegion(trust_radius=1/4), verbose=OPTIMIZE_VERBOSE, report_every=1)
    v1, _ = ep_estimators.get_EP_Estimate(data=data, verbose=OPTIMIZE_VERBOSE, report_every=1)
    assert(v0 > 1.8)
    assert(abs(v0-v1) < 1e-1)

def test_numpy_to_torch():
    x = np.random.randn(100, 10)
    utils.numpy_to_torch(x)
    utils.numpy_to_torch(x > 0)  # bool conversion
    for t in ['float16','int32','int32','float64']:
        utils.numpy_to_torch(x.astype(t))


def test_constructor():
    beta, J, S, F = get_simulation_results()
    i = 2
    g_samples  = observables.get_g_observables(S, F, i)
    rev        = -g_samples
    
    data = observables.Dataset(g_samples=g_samples)

    theta = np.random.rand(data.nobservables)

    data.get_objective(theta)
    data.get_gradient(theta)
    data.get_hessian(theta)
    data.get_objective(theta)
    data.get_tilted_mean(theta)

    observables.Dataset(g_samples=utils.numpy_to_torch(g_samples))
    observables.Dataset(g_samples=utils.numpy_to_torch(g_samples), rev_g_samples=rev)
    observables.Dataset(g_samples=g_samples, rev_g_samples=-utils.numpy_to_torch(rev))
    observables.Dataset(g_samples=utils.numpy_to_torch(g_samples), rev_g_samples=-utils.numpy_to_torch(rev))

def run_inference(beta, J, S, F, num_chunks=None, do_rev=False):
    num_samples_per_spin, N = S.shape
    total_flips = N * num_samples_per_spin  # Total spin-flip attempts

    # Running sums to keep track of EP estimates
    sigma_emp = sigma_g = sigma_gH = sigma_g2 = sigma_g2H = sigma_g3 = sigma_g3H = sigma_N1 = sigma_MTUR = 0.0

    # Because system is multipartite, we can separately estimate EP for each spin
    for i in tqdm(range(N)):
        # frequency of spin i flips
        p_i            =  F[:,i].sum()/(total_flips if total_flips > 0 else 1)

        # Select states in which spin i flipped and use it create object for EP estimation 
        g_samples = observables.get_g_observables(S, F, i)
        data = observables.Dataset(g_samples=g_samples, 
                                     rev_g_samples=-g_samples if do_rev else None,
                                     num_chunks=num_chunks)

        # Empirical estimate 
        J_without_i = np.hstack([J[i,:i], J[i,i+1:]])
        spin_emp = beta * float(utils.numpy_to_torch(J_without_i) @ utils.numpy_to_torch(data.g_mean))
        
        spin_N1, _   = ep_estimators.get_EP_Newton1Step(data, verbose=OPTIMIZE_VERBOSE) # 1-step of Newton
        
        # Full optimization with trust-region Newton method and no holdout 
        optimizer    = optimizers.NewtonMethodTrustRegion(trust_radius=1/4)
        spin_newton, _ = ep_estimators.get_EP_Estimate(data, optimizer=optimizer, verbose=OPTIMIZE_VERBOSE)
        
        # Full optimization with trust-region Newton method and holdout
        np.random.seed(1) # Set seed for holdout reproducibility 
        trn, tst = data.split_train_test()
        spin_newtonH, _ = ep_estimators.get_EP_Estimate(data, optimizer=optimizer, validation=tst, verbose=OPTIMIZE_VERBOSE)

        # Full optimization with TRON
        optimizer2   = optimizers.TRON(trust_radius=1/4)
        spin_newton2, _ = ep_estimators.get_EP_Estimate(data, optimizer=optimizer2, verbose=OPTIMIZE_VERBOSE)
        
        # Full optimization with trust-region Newton method and holdout
        np.random.seed(1) # Set seed for holdout reproducibility 
        trn, tst = data.split_train_test()
        spin_newton2H, _ = ep_estimators.get_EP_Estimate(data, optimizer=optimizer2, validation=tst, verbose=OPTIMIZE_VERBOSE)


        # Full optimization with gradient ascent method , no holdout
        spin_grad, _ = ep_estimators.get_EP_Estimate(data, verbose=OPTIMIZE_VERBOSE)

        # Full optimization with gradient ascent method 
        spin_gradH, _ = ep_estimators.get_EP_Estimate(trn, validation=tst, verbose=OPTIMIZE_VERBOSE)
        spin_gradH, _ = ep_estimators.get_EP_Estimate(trn, test=tst, verbose=OPTIMIZE_VERBOSE)
        spin_gradH, _ = ep_estimators.get_EP_Estimate(trn, validation=tst, test=tst, verbose=OPTIMIZE_VERBOSE)

        # Multidimensional TUR
        spin_MTUR, _ = ep_estimators.get_EP_MTUR(data)
        

        sigma_emp  += p_i * spin_emp
        sigma_g    += p_i * spin_newton
        sigma_gH   += p_i * spin_newtonH
        sigma_N1   += p_i * spin_N1
        sigma_MTUR += p_i * spin_MTUR
        sigma_g2   += p_i * spin_grad
        sigma_g2H  += p_i * spin_gradH
        sigma_g3   += p_i * spin_newton
        sigma_g3H  += p_i * spin_newtonH

        utils.empty_torch_cache()
        return (sigma_emp, sigma_g, sigma_gH, sigma_g2, sigma_g2H, sigma_g3, sigma_g3H, sigma_N1, sigma_MTUR)
    
def test_inference_nochunk():
   # Now testested as part of test_inference_chunks
   beta, J, S, F = get_simulation_results()
   sigma_emp, sigma_g, sigma_gH, sigma_g2, sigma_g2H, sigma_g3, sigma_g3H, sigma_N1, sigma_MTUR = run_inference(beta, J, S, F)
   assert(sigma_emp > .07)
   assert( abs(sigma_gH -sigma_g2H)<2e-2)
   assert( abs(sigma_gH -sigma_g3H)<2e-2)
   assert( abs(sigma_gH -sigma_emp)<2e-2)

def test_inference_empty():
   beta, J, S, F = get_simulation_results()
   run_inference(beta, J, S[:0], F[:0])

def test_inference_chunks():
    beta, J, S, F = get_simulation_results()
    
    results1 = np.array(run_inference(beta, J, S, F))
    for num_chunks in [1, 10]:
        for do_rev in [True, False]:
            results2 = np.array( run_inference(beta, J, S, F, num_chunks=num_chunks, do_rev=do_rev) )
            assert(np.linalg.norm(results1 - results2 )<1e-5)



def test_objective():
    nobservables = 9
    samp = np.random.randn(100,nobservables)
    data = observables.Dataset(g_samples=samp, rev_g_samples=-samp)
    assert(np.isclose( data.get_objective(np.zeros(nobservables)),0))



def test_consistency():


    # N    = 10     # system size
    # k    = 6      # avg number of neighbors in sparse coupling matrix
    # beta = 3.25   # inverse temperature
    # np.random.seed(42) # Set seed for reproducibility

    # J    = spin_model.get_couplings_random(N=N, k=k)
    # S, F = spin_model.run_simulation(beta=beta, J=J, samples_per_spin=1000)

    beta, J, S, F = get_simulation_results()
    sigma_emp = spin_model.get_empirical_EP(beta, J, S, F)

    X0, X1 = spin_model.convert_to_nonmultipartite(S, F)

    N = J.shape[0]
    g_samples   = np.vstack([ X1[:,i]*X0[:,j] - X0[:,i]*X1[:,j] 
                            for i in range(N) for j in range(i+1, N) ]).T
    data1       = observables.Dataset(g_samples=g_samples, rev_g_samples=-g_samples)
    trn1, tst1  = data1.split_train_test()
    sigma_g1, _ = ep_estimators.get_EP_Estimate(trn1, validation=tst1, verbose=OPTIMIZE_VERBOSE)
    assert(sigma_g1 is not None)

    data2       = observables.Dataset(g_samples=g_samples)
    trn2, tst2  = data2.split_train_test()
    sigma_g2, _ = ep_estimators.get_EP_Estimate(trn2, validation=tst2, verbose=OPTIMIZE_VERBOSE)
    assert(sigma_g2 is not None)

    data3       = observables.CrossCorrelations1(X0, X1)
    trn3, tst3  = data3.split_train_test()
    sigma_g3, _ = ep_estimators.get_EP_Estimate(trn3, validation=tst3, verbose=OPTIMIZE_VERBOSE)
    assert(sigma_g3 is not None)

    theta = np.random.rand(data1.nobservables)
    assert((data1.g_mean - data2.g_mean).norm()<1e-5)
    assert(np.abs(data1.get_objective(theta)-data2.get_objective(theta))<1e-5)
    assert((data1.get_tilted_mean(theta)-data2.get_tilted_mean(theta)).norm()<1e-5)

    assert((data1.g_mean - data3.g_mean).norm()<1e-5)
    assert(np.abs(data1.get_objective(theta)-data3.get_objective(theta))<1e-5)
    assert((data1.get_tilted_mean(theta)-data3.get_tilted_mean(theta)).norm()<1e-5)



import optimizers
# Test optimizers
def test_optimizer_base(split=0, max_trn_objective=None, max_val_objective=None, verbose=OPTIMIZE_VERBOSE, max_iter=None):
    beta, J, S, F = get_simulation_results()
    i   = 9   # spin index
    trn = observables.Dataset(observables.get_g_observables(S, F, i))
    val, tst = None, None
    if split == 1:
        trn, val = trn.split_train_test()
    elif split == 2:
        trn, val, tst = trn.split_train_val_test()

    for k in optimizers.OPTIMIZERS.values():
        x0 = np.zeros(trn.nobservables)
        optimizers.optimize(x0=x0, objective=trn, optimizer=k(), max_iter=max_iter, validation=val, verbose=verbose, 
                minimize=False, max_trn_objective=max_trn_objective, max_val_objective=max_val_objective)
    
                                    
def test_optimizer_bounds():
    test_optimizer_base(split=False, max_trn_objective=1e2, max_iter=100)
    test_optimizer_base(split=True, max_trn_objective=1e2, max_val_objective=1e2, max_iter=100)
    test_optimizer_base(split=True, max_trn_objective=1e2, max_val_objective=1e2, max_iter=100)

def test_optimizer_split_verbose():
    test_optimizer_base(verbose=True, max_iter=100)
    test_optimizer_base(split=1, verbose=True, max_iter=100)
    test_optimizer_base(split=2, verbose=True, max_iter=100)


def test_optimizers():
    N    = 100
    J    = spin_model.get_couplings_random(N=N, k=10)
    S, F = spin_model.run_simulation(beta=2, J=J, samples_per_spin=10000)
    i    = 0 # spin index
    g_samples = observables.get_g_observables(S, F, i)


    data = observables.Dataset(g_samples=g_samples)
    x0   = np.zeros(data.nobservables)
    trn2, val2 = data.split_train_test()
    for k in optimizers.OPTIMIZERS.values():
        r = optimizers.optimize(x0=x0, objective=trn2, optimizer=k(), validation=val2, minimize=False, verbose=OPTIMIZE_VERBOSE)
    

class TestObjectiveNumpy(optimizers.Objective):
    y = np.array([1,2,3,4])
    I = np.eye(len(y))
    def get_gradient(self, x):
        return x - self.y

    def get_hessian(self, x):
        return self.I

    def get_objective(self, x):
        return (x - self.y)@(x - self.y)/2

    def split_train_val_test(self):
        return self, None, None


class TestObjectiveTorch(TestObjectiveNumpy):
    y = utils.numpy_to_torch(np.array([1,2,3,4]))
    I = utils.numpy_to_torch(np.eye(len(y)))

    def initialize_parameters(self, x0):
        return utils.numpy_to_torch(x0)
    
    def get_gradient(self, x):
        return x - self.y

    def get_hessian(self, x):
        return self.I

    def get_objective(self, x):
        return (x - self.y)@(x - self.y)/2

    def split_train_val_test(self):
        return self, None, None

class TestObjectiveNumpyNeg(TestObjectiveNumpy):
    def get_objective(self, x):
        return -super().get_objective(x)
    def get_gradient(self, x):
        return -super().get_gradient(x)
    def get_hessian(self, x):   
        return -super().get_hessian(x)

class TestObjectiveTorchNeg(TestObjectiveTorch):
    def get_objective(self, x):
        return -super().get_objective(x)
    def get_gradient(self, x):
        return -super().get_gradient(x)
    def get_hessian(self, x):   
        return -super().get_hessian(x)


def _base_test_optimizers_torch(objective, minimize):
    x0   = objective.y * 0
    for k in optimizers.OPTIMIZERS.values():
        r = optimizers.optimize(x0=x0, 
                                objective=objective, 
                                minimize=minimize, 
                                optimizer=k())
        assert(abs(r.objective) < 1e-3)
        assert( (r.x - objective.y ) @ (r.x - objective.y ) < 1e-4)

    for k in optimizers.OPTIMIZERS.keys():
        r = optimizers.optimize(x0=x0, 
                                objective=objective, 
                                minimize=minimize, 
                                optimizer=k, max_iter=10)

def test_optimizers_numpy_min():
    _base_test_optimizers_torch(TestObjectiveNumpyNeg(), minimize=False)
def test_optimizers_numpy_max():
    _base_test_optimizers_torch(TestObjectiveNumpy(), minimize=True)

def test_optimizers_torch_min():
    _base_test_optimizers_torch(TestObjectiveTorchNeg(), minimize=False)
def test_optimizers_torch_max():
    _base_test_optimizers_torch(TestObjectiveTorch(), minimize=True)
