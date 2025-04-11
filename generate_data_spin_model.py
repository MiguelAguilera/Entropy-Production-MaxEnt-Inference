import os
import argparse
import numpy as np
from spin_model_simulation import run_simulation  # Assumes a custom simulation function

# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Run spin model simulations with varying beta values.")
parser.add_argument("--num_steps", type=int, default=2**7,
                    help="Number of simulation steps (default: 128).")
parser.add_argument("--rep", type=int, default=1_000_000,
                    help="Number of repetitions for the simulation (default: 1000000).")
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

# Flags for update mode: sequential or parallel
parser.add_argument("--sequential", action="store_true", help="Enable sequential update mode.")
parser.add_argument("--parallel", dest="sequential", action="store_false", help="Enable parallel update mode.")
parser.set_defaults(sequential=True)  # Default to sequential mode

# Flags for adding the critical beta
parser.add_argument("--add_critical_beta", action="store_true", help="Add the value of the critical beta to the list.")
parser.set_defaults(add_critical_beta=True)  # Default to sequential mode
parser.add_argument("--critical_beta", type=int, default=1.3484999614126383,
                    help="Value of the critical beta (default: 1.3484999614126383).")


args = parser.parse_args()

# -------------------------------
# Initialization
# -------------------------------
BASE_DIR = os.path.expanduser(args.BASE_DIR)  # Expand user path (e.g., ~)
DTYPE = 'float32'  # Data type used (if relevant in downstream code)

# Simulation parameters
overwrite = True  # Whether to overwrite existing files

# Generate array of beta values
betas = np.linspace(args.beta_min, args.beta_max, args.num_beta)
if args.add_critical_beta:
    betas = np.append(betas, args.critical_beta)
    betas = np.append(args.critical_beta,betas)

BASE_DIR_MODE = BASE_DIR + '/' + ("sequential" if args.sequential else "parallel")
    
# Ensure base directory exists
if not os.path.exists(BASE_DIR_MODE):
    print(f'Creating base directory: {BASE_DIR_MODE}')
    os.makedirs(BASE_DIR_MODE)

# -------------------------------
# Run Simulations
# -------------------------------
for beta in betas:
    beta = np.round(beta, 8)  # Avoid floating-point inconsistencies in filenames

    print(f"\n** Running simulation for system size {args.size} with beta = {beta} **", flush=True)

    # Construct file name based on mode (sequential or parallel)
    file_name = (
        f"{BASE_DIR_MODE}/run_reps_{args.rep}_steps_{args.num_steps}_"
        f"{args.size:06d}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}.h5"
    )

    # Handle file existence
    if os.path.exists(file_name):
        if not overwrite:
            print(f"File {file_name} exists, skipping simulation.")
            continue
        else:
            print(f"File {file_name} exists, overwriting.")
            os.remove(file_name)

    # Run the simulation
    run_simulation(
        file_name=file_name,
        N=args.size,
        num_steps=args.num_steps,
        rep=args.rep,
        beta=beta,
        J0=args.J0,
        DJ=args.DJ,
        seed=42,
        sequential=args.sequential
    )

