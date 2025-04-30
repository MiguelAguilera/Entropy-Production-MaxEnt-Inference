import time, os, argparse
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]='1'

import torch
import utils

utils.set_default_device()

parser = argparse.ArgumentParser(description='Benchmark linear solvers')
    
# Add the --printx argument that stores True when used
parser.add_argument('--printx', action='store_true', default=False, help='Print solution')
    
# Parse the arguments
args = parser.parse_args()
    


def get_A_b():
    n = 1000
    rand_mat = torch.randn(n, n)
    A = rand_mat @ rand_mat.T  # This creates a symmetric PSD matrix
    A += utils.eye_like(A)*1e-4
    b = torch.randn(n)
    return A, b

num_runs = 10

for method in ['solve','solve_ex','steihaug','cholesky','cholesky_ex','QR','lstsq','inv']:
    tot_time = 0
    if args.printx:
        print()
    for i in range(num_runs):
        torch.manual_seed(i)
        A, b = get_A_b()
        kw_args = {} if method != 'steihaug' else dict(trust_radius=10000)
        stime = time.time()
        x = utils.solve_linear_psd(A, b, method=method, **kw_args)
        tot_time += time.time() - stime
        utils.empty_cache()
        if args.printx:
            print(f'{method:15s} {x[:4].cpu().numpy()}')
    print(f'{method:15s} {tot_time:3f}')
