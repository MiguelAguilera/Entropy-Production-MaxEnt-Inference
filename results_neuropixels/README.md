# results_neuropixels

## Overview

This directory contains code used to generate entropy production (EP) results and plots based on Neuropixels recordings from the Allen Brain Observatory. The scripts implement Maximum Entropy (MaxEnt) inference to estimate EP and reproduce key figures from the associated publication.

## How to Reproduce Figure 2

Follow the steps below to regenerate the results and figures shown in Figure 2 of the paper:

### 1. Download and preprocess Neuropixels data

```bash
python get_data_neuropixels.py
```

This script uses the Allen SDK to download and preprocess spike trains into binned activity matrices.

### 2. Estimate and plot entropy production for Figure 2a

```bash
python calculate_Fig2a.py
python display_Fig2a.py
```

- `calculate_Fig2a.py`: Aggregates EP across session types and population sizes.
- `display_Fig2a.py`: Produces the summary plot shown in Figure 2a.

### 3. Estimate and visualize θ matrix for a single session (Figure 2b)

```bash
python calculate_Fig2b.py
```

This script performs MaxEnt fitting for one Neuropixels session and generates the inferred coupling matrix θ used in the visualization of Figure 2b.

## Notes

- Results are saved in `ep_data/` and `ep_fit_results/` directories.
- Figures are saved in `img/`.
- Dependencies can be installed via:

```bash
pip install -r requirements.txt
