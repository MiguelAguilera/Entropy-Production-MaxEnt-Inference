# results_spin_model

## Overview

This directory contains code for generating and analyzing entropy production (EP) in synthetic spin model systems. These controlled simulations serve as benchmarks for validating the MaxEnt EP inference methods under known dynamics.

## How to Reproduce Figure 1

Follow the steps below to reproduce the results and figures from spin model simulations featured in Figure 1 of the paper.

### 1. Generate spin model data

```bash
python generate_data_spin_model.py
```

Simulates spin trajectories across a range of inverse temperatures (beta values) and stores the results in compressed `.npz` format.

### 2. Estimate EP for spin simulations




### 3. Calculate and plot EP results

```bash
python calculate_Fig1a.py
python calculate_Fig1b.py
```

- `calculate_Fig1a.py`: Performs MaxEnt-based EP inference on the generated spin data across different beta values (Figure 1a).
- `calculate_Fig1b.py`: Displays the inferred parameters for the MaxEnt estimator (Figure 1b).

## Notes

- Results are stored in `ep_data/` as `.h5` files containing EP estimates.
- Simulation inputs and metadata are saved in `MaxEntData/` as `.npz`.
- Dependencies can be installed with:

```bash
pip install -r requirements.txt
```

- Visualizations are automatically saved in `img/` if enabled within the scripts.
