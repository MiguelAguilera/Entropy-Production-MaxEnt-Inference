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

kappa = 0.9

vals_ours = []
vals_NEEP = []
vals_ep   = []

logZs = []

ANTISYMMETRIC_OBSERVABLE = False

if True:  # sweep across driving strengths (asymmetry parameters)
    SEMILOG=False
    XLABEL = r'Driving strength $\lambda$'
    #l_vals = np.logspace(-20,0,100)
    l_vals = np.linspace(0, 20, 100, endpoint=True)
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

        ep_ours, theta = f(P, R, obs, method=0)
        ep_NEEP, _ = f(P, R, obs, method=1)
        vals_ours.append(ep_ours)
        vals_NEEP.append(ep_NEEP)
        vals_ep.append( kl(P,R) )
        logZs.append( np.log(np.sum(R*np.exp(obs*theta))))


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
sns.set(style='white', font_scale=1.4)
plt.rc('text', usetex=True)
plt.rc('font', size=14, family='serif', serif=['latin modern roman'])
#plt.rc('legend', fontsize=12)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm,newtxtext}')

cmap = plt.get_cmap('inferno_r')

plt.figure(figsize=(4.5,3.25), layout='constrained')
#var_bound = np.log(27)/2-1
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

plt.legend()
plt.xlabel(XLABEL)
plt.xlim(l_vals.min(), l_vals.max())
plt.ylim(0, 1.1*max(vals_ep.max(), vals_ours.max(), vals_NEEP.max()))
plt.ylabel('EP', rotation=0, labelpad=15)

fname = 'img/vs-neep' + ('-as' if ANTISYMMETRIC_OBSERVABLE else '')
plt.savefig(fname + '.pdf', bbox_inches='tight', pad_inches=0.1)
import os
os.system('pdfcrop ' + fname)

plt.show()


if False:
    plt.figure(figsize=(4,3))
    plt.plot(l_vals, logZs)
    plt.ylabel(r'$\ln Z$', rotation=0, labelpad=15)
    plt.xlabel(XLABEL)
    plt.savefig('img/vs-neep-logZ.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()


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
