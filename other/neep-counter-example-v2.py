import numpy as np
import scipy.special as slp

def kl(p,q): # KL divergence
    return np.sum(slp.kl_div(p,q))

def get_P(a,b):
    # construct a 3x3 steady-state joint probability matrix
    # Here we consider a uniform unicyclic system, with backward rate r
    d = 1-a-b # probability of staying on same state
    assert(d >= -1e-10)
    if d <= 0: 
        d=0
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

    assert(np.allclose(P.sum(axis=0), 1/3))  
    assert(np.allclose(P.sum(axis=1), 1/3))  

    return P, R

from scipy.optimize import minimize_scalar

def dv_neep_objectives(P, R, obs, method, bounds=(-50, 50)):
    """
    Optimize DV or NEEP objective using SciPy.
    
    P, R: probability distributions (1D arrays, sum to 1)
    obs: observation values (same length as P, R)
    method: 0 = DV, 1 = NEEP
    """
    
    expectation = np.sum(P * obs)
    
    def dv_obj(x):
        return -(x * expectation - np.log(np.sum(R * np.exp(x * obs))))
    
    def neep_obj(x):
        return -(x * expectation - np.sum(R * np.exp(x * obs)) + 1)
    
    obj = dv_obj if method == 0 else neep_obj
    
    res = minimize_scalar(obj, method="bounded", bounds=bounds)
    #print(method, res.x, 'dv:', -dv_obj(res.x), 'neep:', -neep_obj(res.x))
    return -res.fun, res.x


# Create observables
M =np.array([[ 0, 1, 0],
             [ 0, 0, 1],
             [ 1, 0, 0]], dtype='float') 
A = M-M.T  # antisymmetric part
S = M+M.T  # symmetric     part

kappa = 0.9
MAX_LAMBDA = 20

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', font_scale=1.3)
plt.rc('text', usetex=True)
plt.rc('font', size=14, family='serif', serif=['latin modern roman'])
#plt.rc('legend', fontsize=12)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm,newtxtext,newtxmath}')

cmap = plt.get_cmap('inferno_r')

plt.figure(figsize=(5,2.5), layout='constrained')
#var_bound = np.log(27)/2-1

for sndx, ANTISYMMETRIC_OBSERVABLE in enumerate([True,False]):

    vals_ours = []
    vals_NEEP = []
    vals_ep   = []

    logZs = []

    SEMILOG=False
    XLABEL = r'Driving strength $\lambda$'
    #l_vals = np.logspace(-20,0,100)
    l_vals = np.linspace(0, MAX_LAMBDA, 100, endpoint=True)
    for l in l_vals:
        r=np.exp(-l)
        a=1/(1+r)*kappa
        b=r/(1+r)*kappa
        P, R = get_P(a,b)
        
        if ANTISYMMETRIC_OBSERVABLE:
            obs = A.copy()
            obs[1,0] =  .2
            obs[0,1] = -.2
            assert(np.allclose(A,-A.T))
        else:
            obs = A + 1

        if False:
            ep_ours, theta = f(P, R, obs, method=0)
            ep_NEEP, _ = f(P, R, obs, method=1)
        else:
            ep_ours = dv_neep_objectives(P, R, obs, 0)[0]
            ep_NEEP = dv_neep_objectives(P, R, obs, 1)[0]
            assert(ep_ours >= ep_NEEP-1e-5)

        vals_ours.append(ep_ours)
        vals_NEEP.append(ep_NEEP)
        vals_ep.append( kl(P,R) )

        # logZs.append( np.log(np.sum(R*np.exp(obs*theta))))



    vals_ours = np.array(vals_ours)
    vals_NEEP = np.array(vals_NEEP)
    vals_ep   = np.array(vals_ep)

    plt.subplot(1,2,sndx + 1)
    var_bound = (5*np.log(5)-4)*kappa
    var_bound = -2*kappa + 2*(1+kappa)*np.arctanh(kappa)
    print(var_bound)
    plt.plot(l_vals, vals_ep, 'k', linestyle=(0, (2, 3)), lw=3, label=r'$\Sigma$', zorder=10)

    plt.plot(l_vals, vals_ours, label=r'$\Sigma_g$', c=cmap(0.75))
    if SEMILOG:
        plt.semilogx()
    plt.plot(l_vals, vals_NEEP, label=r'$\Sigma_g^{\text{KO}}$', c=cmap(0.45))
    plt.plot(l_vals, l_vals*0+var_bound, c='k',ls=':'),
    #         label=r'$\Sigma_g^\prime \;\;(\lambda \to \infty)$')

    plt.xlabel(XLABEL)
    plt.xlim(l_vals.min(), l_vals.max())
    plt.ylim(0, 1.1*max(vals_ep.max(), vals_ours.max(), vals_NEEP.max()))

    if sndx == 0:
        plt.legend(handlelength=1.2)
        plt.ylabel('EP', rotation=0, labelpad=15)

    if ANTISYMMETRIC_OBSERVABLE:
        plt.title('Antisymmetric')
    else:
        plt.title('Non-antisymmetric')

fname = 'img/vs-neep-v2'
plt.savefig(fname + '.pdf', bbox_inches='tight', pad_inches=0.1)
import os
os.system('pdfcrop ' + fname)

plt.show()


# if False:
#     plt.figure(figsize=(4,3))
#     plt.plot(l_vals, logZs)
#     plt.ylabel(r'$\ln Z$', rotation=0, labelpad=15)
#     plt.xlabel(XLABEL)
#     plt.savefig('img/vs-neep-logZ.pdf', bbox_inches='tight', pad_inches=0.1)
#     plt.show()


# # Mathematica code to solve optimization

# f[\[Theta]_, \[Alpha]_, \[Beta]_] := \[Theta]  (5  \[Alpha] + \
# \[Beta])/2 + \[Alpha] + \[Beta] - \[Beta]  Exp[
#     5  \[Theta]/2] - \[Alpha]  Exp[\[Theta]/2]
# v = kk;
# (*Solve for \[Theta] that maximizes f*)
# sol = Solve[D[f[\[Theta], v, 0], \[Theta]] == 0, \[Theta], Reals]

# (*Evaluate the maximum value*)
# maxValue = FullSimplify[f[\[Theta], v, 0] /. sol]
# N[maxValue]
