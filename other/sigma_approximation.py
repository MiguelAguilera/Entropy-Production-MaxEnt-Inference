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

l=2
r=np.exp(-l)
a=1/(1+r)*.9
b=r/(1+r)*.9
P, R = get_P(a,b)
obs = A + 1.5*S

epg, theta = f(P, R, obs, method=0)

print('Average EP, Σ:', kl(P,R))
print('Estimated EP, Σ_g:', np.sum(P*obs)*theta - np.log(np.sum(R*np.exp(obs*theta))))

print(r'values of σ(x):')
print(np.log(P/R))
print(r'approximation σ(x) ≈ θ·x:')
print(obs*theta)
print(r'approximation σ(x) ≈ θ·x - ln(Z):')
print(obs*theta - np.log(np.sum(R*np.exp(obs*theta))))

sigma=np.log(P/R)
sigma1=obs*theta
sigma2=obs*theta - np.log(np.sum(R*np.exp(obs*theta)))

P_nodiag = P.copy()
P_nodiag[range(3),range(3)]=0


print('Mean squared errors          :', np.sum(P*(sigma-sigma1)**2), np.sum(P*(sigma-sigma2)**2))
print('Mean squared errors (no diag):', np.sum(P_nodiag*(sigma-sigma1)**2), np.sum(P_nodiag*(sigma-sigma2)**2))


N=1001
X=np.linspace(0,2,N)
e1=np.zeros(N)
e2=np.zeros(N)
for i,x in enumerate(X):
    sigmax=obs*theta - x*np.log(np.sum(R*np.exp(obs*theta)))
    e1[i] = np.sum(P*(sigma-sigmax)**2)
    e2[i] = np.sum(P_nodiag*(sigma-sigmax)**2)



import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', font_scale=1)
plt.rc('text', usetex=True)
plt.rc('font', size=14, family='serif', serif=['latin modern roman'])
#plt.rc('legend', fontsize=12)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm,newtxtext}')

cmap = plt.get_cmap('inferno_r')
colors = [cmap(0.5), cmap(0.75)]

i1=np.argmin(e1)
i2=np.argmin(e2)

plt.figure()
plt.plot(X,e1, color=colors[0], label='MSE')
plt.plot(X[i1],e1[i1], '*', color=colors[0])
plt.plot(X,e2, color=colors[1], label='MSE (no diag)')
plt.plot(X[i2],e2[i2], '*', color=colors[1])
plt.xlabel(r'$\gamma$,\quad ($\sigma(x) \approx \theta g(x) - \gamma \ln Z$)') 
plt.legend()
plt.show()
