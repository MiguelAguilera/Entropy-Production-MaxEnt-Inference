import numpy as np
import torch
import spin_model
from methods_EP_multipartite import *

J, H, S, F = spin_model.run_simulation(N=10, beta=4, seed=42, sequential=True)
assert(np.all(H==0))  # We do not support local fields in our analysis


J_t = torch.from_numpy(J)

# Initialize accumulators
S_Emp = S_TUR = S_N1 = S_N2 = 0

N, rep = S.shape
T = N * rep  # Total spin-flip attempts

for i in range(N):
    idxs = np.where(F[i, :] == 1)[0]
    S_i_t = torch.from_numpy(S[:, idxs].astype('float32'))


    # Estimate entropy production using various methods
    sig_N1, sig_MTUR, theta1, Da = get_EP_Newton(S_i_t, T, i)
    sigma_emp                    = exp_EP_spin_model(Da, J_t, i)
    sig_N2, theta2               = get_EP_Newton2(S_i_t, T, theta1, Da, i)

    # Aggregate results
    S_Emp += sigma_emp
    S_TUR += sig_MTUR
    S_N1  += sig_N1
    S_N2  += sig_N2

print("\n[Results]")
print(f"  EP (Empirical)    :    {S_Emp:.6f}")
print(f"  EP (MTUR):             {S_TUR:.6f}")
print(f"  EP (1-step Newton):    {S_N1:.6f}")
print(f"  EP (2-step Newton):    {S_N2:.6f}")
print("-" * 70)

