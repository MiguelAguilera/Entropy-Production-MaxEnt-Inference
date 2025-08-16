#!/usr/bin/env python3

import numpy as np
from scipy.special import rel_entr

def obs(P):
    # calculate average fraction of time spent in state 0
    # note that this is a symmetric observable
    i = 0
    return (np.sum(P[i, :, :])+np.sum(P[:, i, :])+np.sum(P[:, :, i]))/3

def f(i,j,k):  # deterministic coarse-graining map over trajectories (i,j,k): we decimate state state 1
    a = i if i != 1 else 0
    b = j if j != 1 else a
    c = k if k != 1 else b
    return a,b,c

def cg_prob(P):
    P_cg = np.zeros((n_states, n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            for k in range(n_states):
                a,b,c = f(i,j,k)
                P_cg[a,b,c] += P[i,j,k]
    return P_cg



n_states = 3
# Define small uniform unicyclic system, with small backward rate r
r = 1e-5
T = np.array([
    [0.1, 0.9-r, r],
    [r, 0.1, 0.9-r],
    [0.9-r, r, 0.1],
])
pi = np.ones(3)/3 # uniform steady state

# 2-step joint probabilities under forward and backward process
P     = np.zeros((n_states, n_states, n_states))
P_rev = np.zeros((n_states, n_states, n_states))
for i in range(n_states):
    for j in range(n_states):
        for k in range(n_states):
            P[i,j,k]     = pi[i] * T[i,j] * T[j,k]
            P_rev[i,j,k] = pi[k] * T[k,j] * T[j,i]

def kl(p, q):  # KL divergence
    return rel_entr(p,q).sum()

epr_full = kl(P, P_rev)
print(f"\nEPR (full system): {epr_full:.6f}")



print(f"avg # steps in state 0 (forward   ): {obs(P):.6f}")
print(f"avg # steps in state 0 (reverse   ): {obs(P_rev):.6f}")


P_cg     = cg_prob(P)
P_rev_cg = cg_prob(P_rev)

epr_cg = kl(P_cg, P_rev_cg)
print(f"\nEPR (coarse-grained): {epr_cg:.6f}")

print(f"avg # steps in state 0 (forward-cg): {obs(P_cg):.6f}")
print(f"avg # steps in state 0 (reverse-cg): {obs(P_rev_cg):.6f}")

