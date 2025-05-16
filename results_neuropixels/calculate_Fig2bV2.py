import argparse, sys, os
import numpy as np

from pathlib import Path
import h5py
import hdf5plugin

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import torch


sys.path.insert(0, '..')
# from methods_EP_parallel import *
import ep_estimators_bak as ep_estimators
import utils
utils.set_default_torch_device()
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description="Estimate EP and MaxEnt parameters for a neuropixels dataset.")
parser.add_argument("--BASE_DIR", type=str, default="~/Neuropixels",
                    help="Base directory to store the data (default: '~/Neuropixels').")
parser.add_argument("--obs", type=int, default=1,
                    help="Observable (default: 1).")
args = parser.parse_args()


# --- Constants and Configuration ---
BASE_DIR = Path(args.BASE_DIR).expanduser()
DTYPE = 'float32'
tol_per_param = 1e-6
bin_size = 0.01
session_id = 0 # 8
session_type = 'active'
N = 200  # Number of neurons





print(f"** DOING SYSTEM SIZE {N} of type {session_type} with session ID {session_id} **", flush=True)

# --- Load and preprocess data ---
filename = BASE_DIR / f"data_binsize_{bin_size}_session_{session_id}.h5"
print(f'Loading data from {filename}, session_type={session_type}')

with h5py.File(filename, 'r') as f:
    if session_type == 'active':
        S = f['S_active'][:]
    elif session_type == 'passive':
        S = f['S_passive'][:]
    elif session_type == 'gabor':
        S = f['S_gabor'][:]
    areas = f['areas'][:].astype(str)  # decode bytes to UTF-8

# Select only visual areas (e.g., V1, V2, etc.)
indices = np.where(np.char.startswith(areas, 'V'))[0]
S = S[indices, :]
areas = areas[indices]

# Sort neurons by overall activity
inds = np.argsort(-np.sum(S, axis=1))
S = S[inds[:N], :].astype(DTYPE) * 2. - 1.
areas = areas[inds[:N]]

# Sort neurons by area name
inds2 = np.argsort(areas)
S = S[inds2, :]
areas = areas[inds2]

# --- Fit MaxEnt model ---
#f = lambda theta: -obj_fn(theta, S_t, S1_t)

# With gradient ascent (regular , no Adam), I get
# get_EP_GradientAscent : iteration  1460 | 0.255515s/iter | f_cur_trn= 0.585399 f_cur_tst= 0.330752
# get_EP_GradientAscent : iteration  1470 | 0.255177s/iter | f_cur_trn= 0.587430 f_cur_tst= 0.330722
# get_EP_GradientAscent : [Stopping] Test objective did not improve  (f_new_tst <= f_cur_tst and)  for 30 steps iter 1477
# EP: 0.3307626247406006
# 2025-05-07 21:42:21.722 python[96973:118866165] +[IMKClient subclass]: chose IMKClient_Modern


# > Processing system size 200 neurons
#   [Info] Estimating EP (nsamples: 360322, lr: 0.000500, patience: 30, % with transition: 0.998449)...
#   [Result took 22.283140] EP tst/full: 0.24623 0.48643 | R: 18.40379 | EP tst/R: 0.01338



patience =100
tol=0
batch_size=None
theta_init = None
max_iter = None 
lr=None
use_BB = False
use_Adam = False
if False:
    use_Adam=True
    lr=0.000001

else:
    use_BB = True
    # lr=0.002
    patience =1000
    lr=0.25/N
   # lr=.7/N
  #  lr=0.1/N
    #lr=0.1/N
    # batch_size=5000
    patience=50
   # max_iter = 200

if args.obs == 1:
    cls = ep_estimators.RawDataset
elif args.obs == 2:
    cls = ep_estimators.RawDataset2
else:
    raise ValueError(f"Invalid observable type {args.obs}. Use 1 or 2.")

data = cls(X0=S[:, 1:].T, X1=S[:, :-1].T)
trn, tst = data.split_train_test()

#theta_init = np.random.randn(data.nobservables)/np.sqrt(data.nobservables)

res=ep_estimators.get_EP_GradientAscent(data=trn, validation_data=tst, lr=lr, use_Adam=use_Adam, use_BB=use_BB, # skip_warm_up=True,
                                         tol=tol, verbose=2, report_every=10, patience=patience, batch_size=batch_size,
                                         max_iter=max_iter, theta_init=theta_init)
sigma = res.objective
print("EP:", sigma)

# Extract coupling matrix θ
theta = utils.torch_to_numpy(res.theta)

triu_indices = np.triu_indices(N, k=1)
if args.obs == 2:
    th = theta.reshape(N, N)
else:
    # Upper triangular matrix
    th = np.zeros((N, N), dtype=theta.dtype)
    th[triu_indices[0], triu_indices[1]] = theta
    th=th-th.T

if False and args.obs == 1:
    #asymm = th - th.T

    #data1 = ep_estimators.RawDataset(X0=tst.X0, X1=tst.X1)

    data2 = ep_estimators.RawDataset2(X0=tst.X0, X1=tst.X1)
    print(data2.get_objective(np.reshape(th, [1,-1])))
    
    




filename = f"ep_data/coupling_coefficients_N{N}_obs{args.obs}.npz"

if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))
np.savez(
    filename,
    areas=areas,
    th=th,
    sigma=sigma
)
print(f"Inferred coupling coefficients saved to {filename}")
