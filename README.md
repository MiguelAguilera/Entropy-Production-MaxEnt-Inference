# Inferring Entropy Production in Many-Body Systems Using Nonequilibrium MaxEnt

The repository includes the following files:

**multipartite_example.py**: simple demonstration script, illustrating how to generate data and estimate EP for a multipartite system (the nonequilibrium spin model).

**spin_model.py**: functions related to the nonequilibrium spin model. This includes generation of random coupling matrices, Monte Carlo sampling (`spin_model.run_simulation(...)`), calculation of observables $g_{ij}(\vec{\boldsymbol{x}}) =(x_{i,1}-x_{i,0})x_{j,0}$, and estimation of ``ground-truth'' EP from empirical statistics.

 for the nonequilibrium spin model. We use Numba to accelerate the Monte Carlo code. In order to minimize memory/storage, `spin_model.run_simulation` returns samples in a compressed format that exploits the multipartite nature of the dynamics. Specifically, it returns a pair of numpy arrays `S` and `F` where:
* `S` is a 2-dimensional array of type `int8` containing states sampled from the steady-state distribution. 
For a system with `N` spins, `S` has dimensions `[samples_per_spin] x [N]` and its entries are all -1s and 1s.
* `F` is a 2-dimensional array of type `bool` also with dimensions `[samples_per_spin] x [N]`. `F` provides
a compressed representation of `samples_per_spin * N` samples of spin flips. 
Specifically, `F[j,i]=True` if spin `i` flipped in a random sample starting from state `S[j,:]`, 
and `F[j,i]=False` if it did not flip.
The set of states in which spin `i` flipped can be accessed as `S[F[:,i],:]`.

**ep_multipartite.py**: functions to estimate EP from sampled data based on observables , as described in the main text.
We use PyTorch to accelerate estimation/optimization. The EP estimators exploit the multipartite nature of the dynamics. The estimators
are implemented as methods of an object created as `ep_multipartite.EPEstimators(S_i, i)`. Here `S_i` is a 2d array of states in which spin `i`
was observed to flip and `i` is the spin index.  For instance, for data generated from the spin model, `S_i` can be computed as `S_i=S[F[:,i],:]`

**results_spinmodel** and **results_neuropixels**: directories containing code for the two examples discussed in the manuscript.
