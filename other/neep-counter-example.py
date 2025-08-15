import numpy as np
import scipy.special as slp
import cvxpy as cp

def kl(p,q): # KL divergence
    return np.sum(slp.kl_div(p,q))



def get_P(a,b):
    # construct a 3x3 steady-state joint probability matrix
    # Here we consider a uniform unicyclic system, with backward rate r
    d = 1-a-b # probability of staying on same state
    assert(d >= 0)
    T = np.array([
        [d, a, b],
        [b, d, a],
        [a, b, d],
    ])
    #print(T[1,0],a,b)
    #sadf
    pi = np.ones(3)/3 # uniform steady state
    assert(np.allclose(T.sum(axis=1), 1.0))  
    assert(np.allclose(pi,pi@T))

    # Forward and reverse probability distributions
    P = T * pi[:,None]
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
    return prob.solve(solver=cp.CLARABEL)

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
    #l_vals = np.logspace(-20,0,100)
    l_vals = np.arange(2, 30)
    for l in l_vals:
        r=np.exp(-l)
        a=1/(1+r)/2
        b=r/(1+r)/2
        P, R = get_P(a,b)
        obs = A + 1.5*S
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
import seaborn as sns
sns.set(style='white', font_scale=1.2)
plt.rc('text', usetex=True)
plt.rc('font', size=14, family='serif', serif=['latin modern roman'])
plt.rc('legend', fontsize=12)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm,newtxtext}')


plt.figure(figsize=(4,3), layout='constrained')
var_bound = np.log(27)/2-1
var_bound = 5*np.log(5)/2-2

plt.plot(l_vals, vals_ep, label=r'$\Sigma$', c='k')

plt.plot(l_vals, vals_ours, label=r'$\Sigma_g$')
if SEMILOG:
    plt.semilogx()
plt.plot(l_vals, vals_NEEP, label=r'$\Sigma_g^\prime$')
plt.plot(l_vals, l_vals*0+var_bound, c='k',ls=':', #ls='none',marker='o', lw=1,markersize=3,
         label=r'$\Sigma_g^\prime \;\;(\lambda \to \infty)$')
plt.legend()
plt.xlabel(XLABEL)
plt.xlim(l_vals.min(), l_vals.max())
plt.ylabel('Entropy production')
plt.savefig('vs-neep.pdf')

import os
os.system('pdfcrop vs-neep')
