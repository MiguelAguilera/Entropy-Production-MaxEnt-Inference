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
    ep = prob.solve(solver=cp.CLARABEL)
    return ep, x.value

# Create observables
M =np.array([[ 0, 1, 0],
             [ 0, 0, 1],
             [ 1, 0, 0]], dtype='float') 
A = M-M.T  # antisymmetric part
S = M+M.T  # symmetric     part


vals_ours = []
vals_NEEP = []
vals_ep   = []

l=4
r=np.exp(-l)
a=1/(1+r)*.9
b=r/(1+r)*.9
P, R = get_P(a,b)
obs = A + 1.5*S

ep, theta = f(P, R, obs, method=0)

print(ep,  kl(P,R))

print(np.sum(P*obs)*theta - np.log(np.sum(R*np.exp(obs*theta))))

print(r'values of σ(x):')
print(np.log(P/R))
print(r'approximation σ(x) ≈ θ·x:')
print(obs*theta)
print(r'approximation σ(x) ≈ θ·x-ln(Z):')
print(obs*theta - np.log(np.sum(R*np.exp(obs*theta))))

sigma=np.log(P/R)
sigma1=obs*theta
sigma2=obs*theta - np.log(np.sum(R*np.exp(obs*theta)))

P_nodiag = P.copy()
P_nodiag[range(3),range(3)]=0

print('Mean squared errors          :', np.sum(P*(sigma-sigma1)**2), np.sum(P*(sigma-sigma2)**2))
print('Mean squared errors (no diag):', np.sum(P_nodiag*(sigma-sigma1)**2), np.sum(P_nodiag*(sigma-sigma2)**2))

