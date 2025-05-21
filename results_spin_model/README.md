# results_spin_model

## Overview

This directory contains all scripts and tools required to reproduce the experiments for the nonequilibrium spin model, as described in the paper:

> Miguel Aguilera, Sosuke Ito, and Artemy Kolchinsky, *Inferring entropy production in many-body systems using nonequilibrium MaxEnt*, 2024.

The spin model experiments demonstrate how entropy production (EP) can be inferred using the nonequilibrium MaxEnt principle. This includes estimation of trajectory-level EP and lower bounds using time-lagged observables in a disordered spin network.

The main contributions include:

- Demonstration of MaxEnt-based EP estimation in a 1000-spin disordered nonequilibrium Ising model.
- Accurate recovery of ground truth EP using time-lagged correlations as observables.
- Validation of the dual formulation through Barzilai–Borwein gradient optimization.

## Reproducing Results (Figure 1 in the paper)

The following steps will reproduce the results in **Fig. 1a** and **Fig. 1b** of the main manuscript.

### 1. Generate synthetic data from the spin model

```bash
python generate_data_spin_model.py
```

This simulates stochastic dynamics in a network of binary spins with asymmetric couplings and saves time series of system states at various inverse temperatures $\beta$.

### 2. Compute entropy production using MaxEnt inference

```bash
python calculate_Fig1a.py
```

This script iterates over $\beta$ values and infers EP using time-lagged spin-spin correlations as observables:

$$
    g_{ij}(\tau) = (x_{i,1} - x_{i,0}) x_{j,0}
$$

Estimates are calculated using the convex dual form of the MaxEnt problem:

$$
    \Sigma_{\bm g} = \max_{\theta} \left[ \theta^T \langle \bm g \rangle_p - \ln \langle e^{\theta^T \bm g} \rangle_{\tilde p} \right]
$$

with optimization performed via gradient ascent with Barzilai–Borwein step sizes.

### 3. Compute and visualize inferred parameters

```bash
python calculate_Fig1b.py
```

Generates the EP estimate plot (Fig. 1a) and a comparison between inferred parameters $\theta^*_{ij} - \theta^*_{ji}$ and true model asymmetries $w_{ij} - w_{ji}$ (Fig. 1b), validating the quality of inference.

---

## File Descriptions

- **generate_data_spin_model.py** — Generates time series from a nonequilibrium Ising model.
- **calculate_Fig1a.py** — Computes EP estimates for multiple $\beta$ values and stores results.
- **calculate_Fig1b.py** — Plots EP values and inferred coupling asymmetries.
- **get_spin_EP.py** — Main interface for estimating EP using the MaxEnt dual formulation.

---

## Output

- **ep_data/** — Contains `.npz` files with system trajectories and `.h5` files with EP estimates.
- **img/Fig1a.pdf** — Plot of true vs. inferred EP for different $\beta$.
- **img/Fig1b.pdf** — Scatter plot of inferred vs. true model asymmetries.

---

## Requirements

Install dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Reference

If you use this code, please cite:

> Aguilera, M., Ito, S., & Kolchinsky, A. (2024). Inferring entropy production in many-body systems using nonequilibrium MaxEnt.

