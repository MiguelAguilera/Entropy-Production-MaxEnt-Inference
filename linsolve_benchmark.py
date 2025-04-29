import time 
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]='1'

import torch
import utils

utils.set_default_device()
def get_A_b():
    n = 1000
    rand_mat = torch.randn(n, n)
    A = rand_mat @ rand_mat.T  # This creates a symmetric PSD matrix
    A += utils.eye_like(A)*1e-4
    b = torch.randn(n)
    return A, b

num_runs = 10

for method in ['solve','solve_ex','cholesky','cholesky_ex','QR','lstsq','inv']:
    tot_time = 0
    for i in range(num_runs):
        torch.manual_seed(i)
        A, b = get_A_b()
        stime = time.time()
        x = utils.solve_linear_psd(A, b, method=method)
        tot_time += time.time() - stime
        utils.empty_cache()
        #print(x[:4].cpu().numpy())
    print(f'{method:15s} {tot_time:3f}')
