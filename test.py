import numpy as np
from numba import njit
import time

@njit('int32(float32, float32[::1], int32[::1])', inline='always')
def GlauberStep(Hi, Ji, s):
    h = Hi + Ji @ s.astype(np.float32)
    return int(np.random.rand() * 2 - 1 < np.tanh(h)) * 2 - 1
    
@njit('int32(float32, float32[::1], int32[::1])', inline='always')
def GlauberStep0(Hi, Ji, s):
    h = Hi + Ji @ s.astype(np.float32)
    return int(0 * 2 - 1 < np.tanh(h)) * 2 - 1
    
# Create random inputs of size 1000
n = 1000
Hi = np.float32(0.1)
Ji = np.random.randn(n).astype(np.float32)


N=100_000
s = np.random.choice([-1, 1], size=(n,N)).astype(np.int32)

h= Ji @ s.astype(np.float32)

# Warm up JIT
GlauberStep(Hi, Ji, s[:,0].copy())
GlauberStep0(Hi, Ji, s[:,0].copy())
@njit
def calc1(N,s):
    for t in range(N):
        si=GlauberStep(Hi, Ji, s[:,t].copy())
    
@njit
def calc2(N,s):
    for t in range(N):
        si=GlauberStep0(Hi, Ji, s[:,t].copy())
    
calc1(1,s[:,0:1])
calc2(1,s[:,0:1])

start = time.time()
calc1(N,s)
end = time.time()
time1 = (end - start)

start = time.time()
calc2(N,s)
end = time.time()
time2 = (end - start)

print(f"Average time per call: {time1:.2f} µs")

print(f"Average time per call: {time2:.2f} µs")

print(time2/time1)



