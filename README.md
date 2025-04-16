# Inferring Entropy Production in Many-Body Systems Using Nonequilibrium MaxEnt

## Nonequilibrium spin model

To generate data for the disordered nonequilibrium spin glass, first run
```
python generate_data_spin_model.py
```
By default, this will draw Monte Carlo samples from $N=100$ spin systems for different inverse temperatures $\beta$, and then save the data to `~/MaxEntData/sequential/`.  Default options (such as system size) can be changed by passing command-line arguments:
```
> python generate_data_spin_model.py --help
usage: generate_data_spin_model.py [-h] [--num_steps NUM_STEPS] [--rep REP]
                                   [--size SIZE] [--BASE_DIR BASE_DIR]
                                   [--beta_min BETA_MIN] [--beta_max BETA_MAX]
                                   [--num_beta NUM_BETA] [--J0 J0] [--DJ DJ]
                                   [--sequential] [--parallel]
                                   [--add_critical_beta]
                                   [--critical_beta CRITICAL_BETA]

Run spin model simulations with varying beta values.

options:
  -h, --help            show this help message and exit
  --num_steps NUM_STEPS
                        Number of simulation steps (default: 128).
  --rep REP             Number of repetitions for the simulation (default:
                        1000000).
  --size SIZE           System size (default: 100).
  --BASE_DIR BASE_DIR   Base directory to store simulation results (default:
                        '~/MaxEntData').
  --beta_min BETA_MIN   Minimum beta value (default: 0).
  --beta_max BETA_MAX   Maximum beta value (default: 4).
  --num_beta NUM_BETA   Number of beta values to simulate (default: 101).
  --J0 J0               Mean interaction coupling (default: 1.0).
  --DJ DJ               Variance of the quenched disorder (default: 0.5).
  --sequential          Enable sequential update mode.
  --parallel            Enable parallel update mode.
  --add_critical_beta   Add the value of the critical beta to the list.
  --critical_beta CRITICAL_BETA
                        Value of the critical beta (default:
                        1.3484999614126383).
```

