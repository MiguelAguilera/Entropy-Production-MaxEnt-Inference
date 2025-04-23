import os, time, argparse
import numpy as np
import h5py
import hdf5plugin
import spin_model


#def save_data(file_name, J, H, S, F):
#    with h5py.File(file_name, 'w') as f:
#        # Save global model parameters
#        f.create_dataset('J', data=J, compression='gzip', compression_opts=5)
#        f.create_dataset('H', data=H, compression='gzip', compression_opts=5)

#        # Save full S and F once
#        f.create_dataset(
#            'S',
#            data=((S + 1) // 2).astype(bool),
#            **hdf5plugin.Blosc(cname='zstd', clevel=4, shuffle=hdf5plugin.Blosc.BITSHUFFLE)
#        )
#        f.create_dataset(
#            'F',
#            data=((F + 1) // 2).astype(bool),
#            **hdf5plugin.Blosc(cname='zstd', clevel=4, shuffle=hdf5plugin.Blosc.BITSHUFFLE)
#        )

#    print(f"Data saved to {file_name}")

def save_data(file_name, J, H, S, F):
    """
    Save model data to a compressed .npz file.

    Parameters:
        file_name (str): Path to the output .npz file.
        J, H : np.ndarray
            Global model parameters.
        S, F : np.ndarray
            Spin states and flip indices.
    """
    # Convert to {0,1} as in original HDF5 version
    S_bin = ((S + 1) // 2).astype(bool)
    F_bin = ((F + 1) // 2).astype(bool)

    # Save all data compressed
    np.savez(
        file_name,
        J=J,
        H=H,
        S=S_bin,
        F=F_bin
    )

    print(f"Compressed data saved to {file_name}")

# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Run spin model simulations with varying beta values.")
parser.add_argument("--num_steps", type=int, default=2**7,
                    help="Number of simulation steps (default: 128).")
parser.add_argument("--rep", type=int, default=1_000_000,
                    help="Number of repetitions for the simulation (default: 1000000).")
parser.add_argument("--trials", type=int, default=1,
                    help="Number of restarts (default: 1).")
parser.add_argument("--size", type=int, default=100,
                    help="System size (default: 100).")
parser.add_argument("--BASE_DIR", type=str, default="~/MaxEntData",
                    help="Base directory to store simulation results (default: '~/MaxEntData').")
parser.add_argument("--beta_min", type=float, default=0,
                    help="Minimum beta value (default: 0).")
parser.add_argument("--beta_max", type=float, default=4,
                    help="Maximum beta value (default: 4).")
parser.add_argument("--num_beta", type=int, default=101,
                    help="Number of beta values to simulate (default: 101).")
parser.add_argument("--J0", type=float, default=1.0,
                    help="Mean interaction coupling (default: 1.0).")
parser.add_argument("--DJ", type=float, default=0.5,
                    help="Variance of the quenched disorder (default: 0.5).")
parser.add_argument("--patterns", type=int, default=None,
                    help="Hopfield pattern density (default: None).")

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

args = parser.parse_args()

# -------------------------------
# Initialization
# -------------------------------
BASE_DIR = os.path.expanduser(args.BASE_DIR)  # Expand user path (e.g., ~)
DTYPE = 'float32'  # Data type used (if relevant in downstream code)

# Simulation parameters
overwrite = False  # Whether to overwrite existing files

# Generate array of beta values
betas = np.linspace(args.beta_min, args.beta_max, args.num_beta)
if args.add_critical_beta:
    betas = np.append(args.critical_beta,betas)

BASE_DIR_MODE = BASE_DIR + '/' + ("sequential" if args.sequential else "parallel")
    
# Ensure base directory exists
if not os.path.exists(BASE_DIR_MODE):
    print(f'Creating base directory: {BASE_DIR_MODE}')
    os.makedirs(BASE_DIR_MODE)

# -------------------------------
# Run Simulations
# -------------------------------
for beta_ix, beta in enumerate(betas):
    beta = np.round(beta, 8)  # Avoid floating-point inconsistencies in filenames

    print(f"\n** Running simulation {beta_ix+1}/{len(betas)} for N={args.size}, β={beta} **", flush=True)

    # Construct file name based on mode (sequential or parallel)
    if args.patterns is None:
        file_name = (
            f"{BASE_DIR_MODE}/run_reps_{args.rep}_steps_{args.num_steps}_"
            f"{args.size:06d}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}.npz"
        )
    else:
        file_name = (
            f"{BASE_DIR_MODE}/run_reps_{args.rep}_steps_{args.num_steps}_"
            f"{args.size:06d}_beta_{beta}_patterns_{args.patterns}.npz"
        )


    # Handle file existence
    if os.path.exists(file_name):
        if not overwrite:
            print(f"File {file_name} exists, skipping simulation.")
            continue
        else:
            print(f"File {file_name} exists, overwriting.")
            os.remove(file_name)

    start_time = time.time()
    J, H, S, F = spin_model.run_simulation(
        N=args.size,
        num_steps=args.num_steps,
        rep=args.rep,
        trials=args.trials,
        beta=beta,
        J0=args.J0,
        DJ=args.DJ,
        seed=args.seed,
        sequential=args.sequential,
        patterns=args.patterns
    )

    print('Sampled states: %d' % S.shape[1])
    print('   - state changes : %d/%d' % ( (F==1).sum(), F.shape[0]*F.shape[1] ) )
    print('   - magnetization : %f' % np.mean(S.astype(float)))
    
    save_data(file_name, J, H, S, F)
    print(f"Simulation took {time.time()-start_time:.3f}s.")


