import torch, time
import utils
utils.set_default_torch_device()

def cov2(X):
     return torch.cov(X.T, correction=0)


def cov1(X, num_chunks=None):
        nsamples     = X.shape[0]
        nobservables = X.shape[1]
        mean         = X.mean(dim=0)
        if num_chunks is None:
            combined_cov = torch.einsum('ki,kj->ij', X, X)

        else:
            # Chunked computation
            combined_cov = torch.zeros((nobservables, nobservables), device=X.device)
            chunk_size = (nsamples + num_chunks - 1) // num_chunks  # Ceiling division

            for r in range(num_chunks):
                start = r * chunk_size
                end = min((r + 1) * chunk_size, nsamples)
                chunk = X[start:end]
                
                combined_cov += torch.einsum('ki,kj->ij', chunk, chunk)

        return combined_cov / nsamples - torch.outer(mean, mean)

with torch.no_grad():
    # Generate random data
    # X = torch.randn(100000, 1000, dtype='float32', device=torch.device('mps'))
    # X = torch.randn(100000, 1000, dtype='float32', device=torch.device('cpu'))
    X = torch.randn(1000000, 1000, device=torch.get_default_device()).to(torch.float32)

    stime = time.time()
    A = cov1(X)
    utils.empty_torch_cache()
    print("Time taken with chunks:", time.time() - stime)


    stime = time.time()
    B = cov2(X)
    utils.empty_torch_cache()
    print("Time taken with cov2:", time.time() - stime)

    print("Are they equal?", torch.allclose(A, B), torch.norm(A-B), A.shape == B.shape)