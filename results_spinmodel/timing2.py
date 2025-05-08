import numpy as np
import torch
import time

# Create a large matrix on CPU
large_matrix = np.random.random((10000, 10000)).astype('float32')

torch.mps.empty_cache()
# Approach 2: Transfer to GPU, then drop column
start = time.time()
gpu_matrix_full = torch.tensor(large_matrix, device='mps')
gpu_matrix_2 = torch.cat([gpu_matrix_full[:, :5000], gpu_matrix_full[:, 5001:]], dim=1)
time_approach_2 = time.time() - start


torch.mps.empty_cache()
# Approach 1: Drop column on CPU, then transfer
start = time.time()
reduced_matrix_cpu = np.delete(large_matrix, 5000, axis=1)  # Delete middle column
gpu_matrix_1 = torch.tensor(reduced_matrix_cpu, device='mps')
time_approach_1 = time.time() - start



print(f"Time for CPU drop then transfer: {time_approach_1:.4f}s")
print(f"Time for transfer then GPU drop: {time_approach_2:.4f}s")
