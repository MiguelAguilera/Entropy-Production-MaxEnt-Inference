import sys, os, time, argparse
import numpy as np

sys.path.insert(0, '..')
import spin_model

def save_data(file_name, J, S, F, beta, args):
    """
    Save model data to a .npz file.

    Arguments:
        file_name (str): Path to the output .npz file.
        J : np.ndarray
            Coupling matrix parameters.
        S : 2d np.array (int8) of {-1,1}
            Indicates -1,1 sampled spin states
        F : 2d np.ndarray (bool)
            Indicates which spins flipped.
    """
    # Spin states S are converted to boolean format, this can sometimes reduce storage requirements
    S_bin = ((S+1)//2).astype(bool)
    np.savez(
        file_name,
        J=J,
        S_bin=S_bin,
        F=F,
        beta=beta, 
        N=args.N,
        DJ=args.DJ,
        J0=args.J0,
        rep=args.rep,
        trials=args.trials,
        seed=args.seed,
    )

    print(f"Data saved to {file_name}")

# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Run spin model simulations with varying beta values.")
parser.add_argument("--rep", type=int, default=1_000_000,
                    help="Number of repetitions for the simulation (default: 1000000).")
parser.add_argument("--trials", type=int, default=1,
                    help="Number of restarts (default: 1).")
parser.add_argument("--N", type=int, default=100,
                    help="System size (default: 100).")
parser.add_argument("--BASE_DIR", type=str, default="~/MaxEntData",
                    help="Base directory to store simulation results (default: '~/MaxEntData').")
parser.add_argument("--beta_min", type=float, default=0,
                    help="Minimum beta value (default: 0).")
parser.add_argument("--beta_max", type=float, default=3,
                    help="Maximum beta value (default: 3).")
parser.add_argument("--num_beta", type=int, default=101,
                    help="Number of beta values to simulate (default: 101).")
parser.add_argument("--J0", type=float, default=0.0,
                    help="Mean interaction coupling (default: 0.0).")
parser.add_argument("--DJ", type=float, default=1.0,
                    help="Variance of the quenched disorder (default: 1.0).")
parser.add_argument("--patterns", type=int, default=None,
                    help="Hopfield pattern density (default: None).")
parser.add_argument("--num_neighbors", type=int, default=None,
                    help="Number of neighbors for sparse connectivity (default: None).")

# Flags for update mode: sequential or parallel
parser.add_argument("--sequential", action="store_true", help="Enable sequential update mode.")
parser.add_argument("--parallel", dest="sequential", action="store_false", help="Enable parallel update mode.")
parser.set_defaults(sequential=True)  # Default to sequential mode

# Flags for adding the critical beta
parser.add_argument("--add_critical_beta", action="store_true", help="Add the value of the critical beta to the list.")
parser.set_defaults(add_critical_beta=False)  # Default to sequential mode
parser.add_argument("--critical_beta", type=float, default=1.3484999614126383,
                    help="Value of the critical beta (default: 1.3484999614126383).")

parser.add_argument("--seed", type=int, default=42,
                    help="Seed for random number generator (negative for no seed) (default: 42).")

parser.add_argument("--overwrite", action="store_true",  default=False, help="Overwrite existing files.")

args = parser.parse_args()

# -------------------------------
# Initialization
# -------------------------------
BASE_DIR = os.path.expanduser(args.BASE_DIR)  # Expand user path (e.g., ~)
DTYPE = 'float32'  # Data type used (if relevant in downstream code)

# Simulation parameters
overwrite = args.overwrite  # Whether to overwrite existing files

# Generate array of beta values
betas = np.linspace(args.beta_min, args.beta_max, args.num_beta)
if args.add_critical_beta:
    betas = np.append(args.critical_beta,betas)

BASE_DIR_MODE = BASE_DIR + '/' + ("sequential" if args.sequential else "parallel")
    
# Ensure base directory exists
if not os.path.exists(BASE_DIR_MODE):
    print(f'Creating base directory: {BASE_DIR_MODE}')
    os.makedirs(BASE_DIR_MODE)

# Create coupling matrix
# Construct file name based on mode (sequential or parallel)
if args.seed >= 0:
    np.random.seed(args.seed) # Seed for reproducibility

if args.patterns is None:
    J = spin_model.get_couplings_random(N=args.N, k=args.num_neighbors, J0=args.J0, DJ=args.DJ)
else:
    J = spin_model.get_couplings_patterns(N=args.N, L=args.patterns)

# -------------------------------
# Run Simulations
# -------------------------------
for beta_ix, beta in enumerate(betas):
    beta = np.round(beta, 8)  # Avoid floating-point inconsistencies in filenames

    print(f"\n# ** Running simulation {beta_ix+1}/{len(betas)} for N={args.N}, β={beta} **", flush=True)

    # Construct file name based on mode (sequential or parallel)
    if args.patterns is None:
        if args.num_neighbors is None:
            file_name = (
                f"{BASE_DIR_MODE}/run_reps_{args.rep}_N_{args.N:06d}_"
                f"beta_{beta}_J0_{args.J0}_DJ_{args.DJ}.npz"
            )
        else:
            file_name = (
                f"{BASE_DIR_MODE}/run_reps_{args.rep}_N_{args.N:06d}_"
                f"beta_{beta}_J0_{args.J0}_DJ_{args.DJ}_num_neighbors_{args.num_neighbors}.npz"
            )
    else:
        file_name = (
            f"{BASE_DIR_MODE}/run_reps_{args.rep}_N_{args.N:06d}_"
            f"beta_{beta}_patterns_{args.patterns}.npz"
        )


    # Handle file existence
    if os.path.exists(file_name):
        if not overwrite:
            print(f"# File {file_name} exists, skipping simulation.")
            continue
        else:
            print(f"# File {file_name} exists, overwriting.")
            os.remove(file_name)

    start_time = time.time()
    S, F = spin_model.run_simulation(
        beta=beta, J=J, samples_per_spin=args.rep, num_restarts=args.trials, sequential=args.sequential,
    )

    print(f'Sampled {S.shape[0]} states, {F.shape[0]*F.shape[1]} transitions, {(F==1).sum()} flips in  {time.time()-start_time:.3f}s')
    save_data(file_name, J, S, F, beta, args)


