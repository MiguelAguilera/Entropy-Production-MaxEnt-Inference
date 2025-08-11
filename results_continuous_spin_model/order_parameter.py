import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

def kuramoto_R_time_series(S):
    """Compute Kuramoto R(t) for each time step."""
    z_t = np.exp(1j * S).mean(axis=1)
    return np.abs(z_t)

def ep_per_spin_timeavg(J, S, S1):
    """
    Compute time-averaged entropy production per spin:
        EP_i(t) = J[i, :] @ ( cos(s1_i(t) - s(t)) - cos(s_i(t) - s(t)) )
    Returns:
        EP_mean_i: (N,) mean over time for each spin
        EP_t_i   : (T, N) per-time per-spin EP values
    """
    T, N = S.shape
    EP_t_i = np.empty((T, N), dtype=S.dtype)
    for i in range(N):
        dC = np.cos(S1[:, [i]] - S) - np.cos(S[:, [i]] - S)
        EP_t_i[:, i] = dC @ J[i, :]
    EP_mean_i = EP_t_i.mean(axis=0)
    return EP_mean_i, EP_t_i

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Kuramoto R and EP vs beta for continuous spin model.")
    parser.add_argument("--rep", type=int, default=1_000_000)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--BASE_DIR", type=str, default="~/MaxEntData")
    parser.add_argument("--beta_min", type=float, default=0)
    parser.add_argument("--beta_max", type=float, default=3)
    parser.add_argument("--num_beta", type=int, default=101)
    parser.add_argument("--J0", type=float, default=0.0)
    parser.add_argument("--DJ", type=float, default=1.0)
    parser.add_argument("--patterns", type=int, default=None)
    parser.add_argument("--num_neighbors", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_plot", action="store_true", default=False)
    parser.add_argument("--mode_dir", type=str, default="cont_sequential")
    args = parser.parse_args()

    N = args.N
    rep = args.rep
    BASE_DIR = os.path.expanduser(args.BASE_DIR)
    betas = np.linspace(args.beta_min, args.beta_max, args.num_beta)

    # Storage
    R_mean = np.full(args.num_beta, np.nan)
    EP_mean_over_spins = np.full(args.num_beta, np.nan)

    for ib, beta in enumerate(np.round(betas, 8)):
        # File path
        if args.patterns is None:
            if args.num_neighbors is None:
                file_name = f"{BASE_DIR}/{args.mode_dir}/run_reps_{rep}_N_{N:06d}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}.npz"
            else:
                file_name = f"{BASE_DIR}/{args.mode_dir}/run_reps_{rep}_N_{N:06d}_beta_{beta}_J0_{args.J0}_DJ_{args.DJ}_num_neighbors_{args.num_neighbors}.npz"
        else:
            file_name = f"{BASE_DIR}/{args.mode_dir}/run_reps_{rep}_N_{N:06d}_beta_{beta}_patterns_{args.patterns}.npz"

        if not os.path.exists(file_name):
            print(f"[Missing] {file_name}")
            continue

        print(f"[Loading] {file_name}")
        data = np.load(file_name)
        if "S" not in data or "S1" not in data or "J" not in data:
            print(f"[Skipping] {file_name} (missing S, S1, or J)")
            continue

        S  = data["S"]
        S1 = data["S1"]
        J  = data["J"]

        # Kuramoto R
        R_t = kuramoto_R_time_series(S)
        R_mean[ib] = R_t.mean()

        # Entropy production per spin
        EP_mean_i, _ = ep_per_spin_timeavg(J, S, S1)
        EP_mean_over_spins[ib] = EP_mean_i.mean()

        print(f"  ⟨R⟩={R_mean[ib]:.4f}  EP/spin={EP_mean_over_spins[ib]:.4g}")

    if not args.no_plot:
        plt.rc('text', usetex=True)
        plt.rc('font', size=18, family='serif', serif=['latin modern roman'])
        plt.rc('legend', fontsize=14)

        # Figure 1: R vs beta
        plt.figure(figsize=(5,4))
        plt.plot(betas, R_mean, 'o-', lw=2, color='C0', label=r'$\langle R\rangle$')
        plt.xlabel(r'$\beta$')
        plt.ylabel(r'$\langle R\rangle$', rotation=0, labelpad=12)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()

        # Figure 2: EP/spin vs beta
        plt.figure(figsize=(5,4))
        plt.plot(betas, EP_mean_over_spins, 's-', lw=2, color='C1', label='EP/spin')
        plt.xlabel(r'$\beta$')
        plt.ylabel('EP/spin', rotation=0, labelpad=12)
        plt.legend()
        plt.tight_layout()

        plt.show()

