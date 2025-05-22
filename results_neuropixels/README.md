# results_neuropixels

## Overview

This directory contains all scripts and tools required to reproduce the experiments based on Neuropixels recordings from the Allen Brain Observatory, as described in the paper:

> Miguel Aguilera, Sosuke Ito, and Artemy Kolchinsky, *Inferring entropy production in many-body systems using nonequilibrium MaxEnt*, 2025.

The Neuropixels experiments demonstrate how entropy production (EP) can be estimated in real neural data using the nonequilibrium MaxEnt principle. The method identifies time-irreversibility in spike-train dynamics and reveals how brain state and task engagement modulate nonequilibrium signatures.

The main contributions include:

- Estimation of entropy production across multiple brain states (active task, passive viewing, and receptive field stimulation).
- Evidence that EP increases with population size and behavioral engagement.
- Visualization of temporal couplings across visual brain areas.

## Reproducing Results (Figure 2 in the paper)

The following steps will reproduce the results in **Fig. 2a** and **Fig. 2b** of the main manuscript.

### 1. Download and preprocess Neuropixels data

```bash
python get_data_neuropixels.py
```

This script prepares spike trains from Allen Brain Observatory recordings and stores them in HDF5 format.

### 2. Estimate EP across sessions and neuron populations

```bash
python calculate_Fig2a.py
```

This script estimates EP across multiple subsets of neurons and conditions using the MaxEnt dual formulation:

$$
    g_{ij}(\tau) = x_{i,1} x_{j,0} - x_{i,0} x_{j,1}
$$

EP is normalized by spike rate and computed across session repetitions. Results are saved for visualization.

### 3. Plot EP scaling across conditions

```bash
python display_Fig2a.py
```

Generates the entropy production scaling plot (Fig. 2a), showing EP per spike as a function of population size for different stimulus/task conditions.

### 4. Visualize inferred coupling structure

```bash
python calculate_Fig2b.py
```

Estimates the interaction matrix \( \theta^*_{ij} \) for a selected session and visualizes its structure grouped by visual brain areas (Fig. 2b).

---

## File Descriptions

- **get_data_neuropixels.py** — Downloads and processes spike train data into binarized activity arrays.
- **calculate_Fig2a.py** — Estimates EP across sessions and neuron population sizes.
- **display_Fig2a.py** — Aggregates and plots EP vs. population size.
- **calculate_Fig2b.py** — Computes and visualizes pairwise MaxEnt parameters for selected neurons.
- **calculate_Fig2bV2.py** — Alternative version of `calculate_Fig2b.py`.
- **fig2.sh** — Optional shell script to generate both Fig. 2a and 2b.

---

## Output

- **ep_data/** — Contains `.h5` files with entropy production estimates for each session and condition.
- **img/Fig2a.pdf** — EP vs. population size for active, passive, and Gabor conditions.
- **img/Fig2b.pdf** — Heatmap of inferred temporal couplings \( \theta^*_{ij} \) across brain regions.

---

## Requirements

Install dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Reference

If you use this code, please cite:

> Aguilera, M., Ito, S., & Kolchinsky, A. (2025). Inferring entropy production in many-body systems using nonequilibrium MaxEnt.
