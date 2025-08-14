import numpy as np
import scipy.special as slp
import cvxpy as cp

def kl(p,q): # KL divergence
    return np.sum(slp.kl_div(p,q))




def get_P(r):
    # construct a 3x3 steady-state joint probability matrix
    # Here we consider a uniform unicyclic system, with backward rate r
    d = .1 # probability of staying on same state
    o = 1-r-d
    assert(o >= 0)
    T = np.array([
        [d, o, r],
        [r, d, o],
        [o, r, d],
    ])
    pi = np.ones(3)/3 # uniform steady state
    assert(np.allclose(T.sum(axis=0), 1.0))  
    assert(np.allclose(pi,T@pi))

    # Forward and reverse probability distributions
    P = T * pi[None,:]
    R = P.T

    return P, R


def f(P, R, obs, method): # Optimize variational expression
    x = cp.Variable()
    expectation = cp.sum(cp.multiply(P, obs))
    if method == 0: # Donsker-Varadhan
        objective = cp.Maximize(x * expectation - cp.log_sum_exp(x * obs + cp.log(R)))
    else:           # NEEP
        objective = cp.Maximize(x * expectation - cp.sum(cp.multiply(R, cp.exp(x * obs))) + 1)
    prob = cp.Problem(objective)
    return prob.solve(solver=cp.MOSEK)

# Create observables
M =np.array([[ 0, 1, 0],
             [ 0, 0, 1],
             [ 1, 0, 0]], dtype='float') 
A = M-M.T  # antisymmetric part
S = M+M.T  # symmetric     part


vals_ours = []
vals_NEEP = []
vals_ep   = []

if True:  # sweep across driving strengths (asymmetry parameters)
    SEMILOG=False
    XLABEL = r'Driving strength $\lambda$'
    l_vals = np.logspace(-20,0,500)/4
    l_vals = np.arange(0, 20)
    for l in l_vals:
        P, R = get_P(r=np.exp(-l)/2)
        obs = A + S * 2
        vals_ours.append( f(P, R, obs, method=0) )
        vals_NEEP.append( f(P, R, obs, method=1) )
        vals_ep.append( kl(P,R) )



else:     # sweep across degrees of asymmetry
    SEMILOG=True
    XLABEL = r'Asymmetric-symmetric mixing coefficient $l$'
    l_vals = np.logspace(-.5,1.5,500)
    P, R = get_P(r=1e-5)
    true_ep = kl(P,R)
    for l in l_vals:
        obs = A + l * S
        vals_ours.append( f(P, R, obs, method=0) )
        vals_NEEP.append( f(P, R, obs, method=1) )
        vals_ep.append(true_ep )


vals_ours = np.array(vals_ours)
vals_NEEP = np.array(vals_NEEP)
vals_ep   = np.array(vals_ep)

import matplotlib.pyplot as plt

plt.plot(l_vals, vals_ep, label='True', c='k')

plt.plot(l_vals, vals_ours, label='Ours')
plt.plot(l_vals, vals_NEEP, label='NEEP')
if SEMILOG:
    plt.semilogx()
plt.legend()
plt.xlabel(XLABEL)
plt.ylabel('Entropy production')
plt.show()
