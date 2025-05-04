# Inferring Entropy Production in Many-Body Systems Using Nonequilibrium MaxEnt

## Nonequilibrium spin model

See `spin_example.py` for a demonstration of how to generate data and estimate EP for the nonequilibrium spin model.
The script calls functions from `spin_model.py` to generate a random coupling matrix and to 
run the Monte Carlo sampler (`spin_model.run_simulation(...)`). It then calls functions from `ep_multipartite.py` to estimate EP from sampled data based on observables $g_{ij}(\vec{\boldsymbol{x}}) =(x_{i,1}-x_{i,0})x_{j,0}$ described in the main text.
For performance, we use `numba` acceleration for sampling and `torch` acceleration for estimation/optimization.


In order to minimize memory/storage, `spin_model.run_simulation` returns samples in a compressed format that exploits the multipartite nature of the dynamics. Specifically, it returns a pair of numpy arrays `S` and `F` where:
* `S` is a 2-dimensional array of type `int8` containing states sampled from the steady-state distribution. 
For a system with `N` spins, `S` has dimensions `[samples_per_spin] x [N]` and its entries are all -1s and 1s.
* `F` is a 2-dimensional array of type `bool` also with dimensions `[samples_per_spin] x [N]`. `F` provides
a compressed representation of `samples_per_spin * N` samples of spin flips. 
Specifically, `F[j,i]=True` if spin `i` flipped in a random sample starting from state `S[j,:]`, 
and `F[j,i]=False` if it did not flip.
The set of states in which spin `i` flipped can be accessed as `S[F[:,i],:]`.

The EP estimators in `ep_multipartite.py` also exploit the multipartite nature of the dynamics. The estimators
are implemented as methods of an object created as `ep_multipartite.EPEstimators(S_i, i)`. Here `S_i=S[F[:,i],:]` is a 2d array of states in which spin `i`
was observed to flip and `i` is the spin index. 
