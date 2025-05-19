# Includes various handy functions
import os
import numpy as np
import warnings

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable fallback for MPS backend
import torch

from config import DTYPE, USE_GPU   # Default data type for torch tensors

# Helpful tensor processing functions


def numpy_to_torch(X, device=None):  # Convert numpy array to torch tensor if needed
    if not isinstance(X, torch.Tensor): 
        if isinstance(X, np.ndarray):
            dtype      = X.dtype
            trg_device = torch.get_default_device() if device is None else device
            if dtype == np.dtype(DTYPE):
                return torch.from_numpy(X              ).to(trg_device).contiguous()
            elif (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_)):
                # Convert on GPU if possible, its a bit faster
                return torch.from_numpy(X              ).to(trg_device).to(getattr(torch, DTYPE)).contiguous()
            else:
                return torch.from_numpy(X.astype(DTYPE)).to(trg_device).contiguous()
        else:
            raise Exception("Argument must be a torch tensor or numpy array")
    else:
        if device is not None:
            return X.to(device)
        else:
            return X


def torch_to_numpy(X):  # Convert torch tensor to numpy array if needed
    if isinstance(X, torch.Tensor):
        return X.numpy(force=True)
    elif isinstance(X, np.ndarray):
        return X
    else:
        raise Exception("Argument must be a torch tensor or numpy array")

def outer_product(a,b):
    # Outer product of two vectors a and b
    return torch.outer(a,b)

# Torch stuff

def set_default_torch_device():
    device = torch.device("cpu")
    if USE_GPU:
        # Determines the best available device for PyTorch operations and sets it as default.
        # Returns the torch device that was set as default ('mps', 'cuda', or 'cpu')
        if torch.backends.mps.is_available():
            if DTYPE != 'float32':
                print(f"MPS backend available, but it only supports float32 data type, not {DTYPE}")
            else:
                device = torch.device("mps")
                warnings.filterwarnings("ignore", message="The operator 'aten::_linalg_solve_ex.result' is not currently supported on the MPS backend and will fall back to run on the CPU", category=UserWarning)
                warnings.filterwarnings("ignore", message="The operator 'aten::triu_indices' is not currently supported on the MPS backend and will fall back to run on the CPU", category=UserWarning)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.set_default_device(device)
    return device


def empty_torch_cache():  # Empty torch cache
    if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
        torch.cuda.empty_cache()
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def torch_synchronize():  # Empty torch cache
    if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
        torch.cuda.synchronize()
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.synchronize()


