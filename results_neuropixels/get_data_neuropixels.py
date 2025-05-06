import numpy as np
from pathlib import Path
import argparse
import h5py
import hdf5plugin
import threading
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache

# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Estimate EP for the spin model with varying beta values.")
parser.add_argument("--BASE_DIR", type=str, default="~/Neuropixels",
                    help="Base directory to store the data (default: '~/Neuropixels').")
args = parser.parse_args()

# Output directory and cache loading
output_dir = Path(args.BASE_DIR).expanduser()
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=output_dir)
cache.load_latest_manifest()

# Load tables from the cache
units_table = cache.get_unit_table()
channels_table = cache.get_channel_table()
probes_table = cache.get_probe_table()
ecephys_sessions_table = cache.get_ecephys_session_table()

# -------------------------------
# Save data function using HDF5
# -------------------------------
def save_data(file_name, areas, S_active, S_passive, S_gabor):
    with h5py.File(file_name, 'a') as f:
        for name, data in zip(['S_active', 'S_passive', 'S_gabor'], [S_active, S_passive, S_gabor]):
            if name in f:
                del f[name]  # Overwrite if already exists
            bool_array = ((data + 1) // 2).astype(bool)
            f.create_dataset(
                name,
                data=bool_array,
                **hdf5plugin.Blosc(cname='zstd', clevel=4, shuffle=hdf5plugin.Blosc.BITSHUFFLE)
            )
        # Save areas as variable-length UTF-8 strings
        if "areas" in f:
            del f["areas"]
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset("areas", data=np.array(areas, dtype=dt))
    print(f"[Thread] Data saved to {file_name}")

# -------------------------------
# Session download and processing
# -------------------------------
def download_session(session, ind):
    """Download and process a session: filter units, bin spikes, extract active/passive epochs."""

    # Merge units and channels to get anatomical info for each unit
    units = session.get_units()
    channels = session.get_channels()
    unit_channels = units.merge(channels, left_on='peak_channel_id', right_index=True)
    unit_channels = unit_channels.sort_values('probe_vertical_position', ascending=False)

    # Filter "good" units based on SNR, ISI violations and firing rate
    good_unit_filter = (
        (unit_channels['snr'] > 1) &
        (unit_channels['isi_violations'] < 1) &
        (unit_channels['firing_rate'] > 0.1)
    )
    good_units = unit_channels.loc[good_unit_filter]

    # Extract spike times and unit IDs
    spike_times = session.spike_times
    unit_ids = good_units.index.tolist()
    areas = good_units['structure_acronym'].tolist()
    N = len(unit_ids)

    print(f"Selected {N} good units")

    if N == 0:
        print("No good units found; skipping session.")
        return

    # Extract stimulus presentations
    stimulus_presentations = session.stimulus_presentations

    # Separate epochs: active, passive (natural images), and gabor
    active_epochs = stimulus_presentations[stimulus_presentations['active'] == True]
    passive_epochs = stimulus_presentations[
        (stimulus_presentations['active'] == False) &
        (stimulus_presentations['image_name'].notna())
    ]
    gabor_epochs = stimulus_presentations[
        (stimulus_presentations['active'] == False) &
        (stimulus_presentations['image_name'].isna())
    ]

    # Compute global min and max spike time across selected units
    maxT = max(np.max(spike_times[uid]) for uid in unit_ids)
    minT = min(np.min(spike_times[uid]) for uid in unit_ids)

    # Discretize spikes into time bins
    bin_size = 0.01  # 10 ms
    T = int(np.floor((maxT - minT) / bin_size))
    
    
    # For tracking bin collisions per unit
    collision_percentages = []

    for u, unit_id in enumerate(unit_ids):
        # Discretize spikes
        spike_times_unit = np.array(spike_times[unit_id])
        spike_times_discretized = np.floor((spike_times_unit - minT) / bin_size).astype(int)

        # Keep only spikes within bounds
        spike_times_discretized = spike_times_discretized[
            (spike_times_discretized >= 0) & (spike_times_discretized < T)
        ]

        # Count how many times each bin is hit
        bincounts = np.bincount(spike_times_discretized, minlength=T)

        # Count how many collisions occurred
        n_collided_spikes = np.sum(bincounts[bincounts > 1] - 1)  # Number of *extra* spikes in bins with >1
        n_total_spikes = len(spike_times_discretized)

        # Collision percentage for this unit
        collision_percentage = (n_collided_spikes / n_total_spikes) * 100 if n_total_spikes > 0 else 0.0
        collision_percentages.append(collision_percentage)

    # Report average bin collision percentage
    mean_collision = np.mean(collision_percentages)
    print(f"Average percentage of spikes lost to bin collisions per unit: {mean_collision:.4f}%")


    S = np.zeros((T,N), dtype=bool)

    for u, unit_id in enumerate(unit_ids):
        spike_times_discretized = np.floor((np.array(spike_times[unit_id]) - minT) / bin_size).astype(int)
        spike_times_discretized = spike_times_discretized[
            (spike_times_discretized >= 0) & (spike_times_discretized < T)
        ]
        S[spike_times_discretized, u] = True

    # Generate masks for different stimulus conditions
    def create_time_mask(epochs):
        mask = np.zeros(T, dtype=bool)
        start_bin = int(np.floor((epochs['start_time'].min() - minT) / bin_size))
        stop_bin = int(np.floor((epochs['end_time'].max() - minT) / bin_size))
        mask[start_bin:stop_bin] = True
        return mask

    active_mask = create_time_mask(active_epochs)
    passive_mask = create_time_mask(passive_epochs)
    gabor_mask = create_time_mask(gabor_epochs)

    print(f"Active bins: {np.sum(active_mask)}, Passive bins: {np.sum(passive_mask)}, Gabor bins: {np.sum(gabor_mask)}")

    # Subset spikes by condition
    S_active = S[active_mask, :]
    S_passive = S[passive_mask, :]
    S_gabor = S[gabor_mask, :]

    # Save in background using HDF5 with Blosc compression
    filename = output_dir / f'data_binsize_{bin_size}_session_{ind}.h5'
    filename.parent.mkdir(parents=True, exist_ok=True)

    t = threading.Thread(target=save_data, args=(filename, areas, S_active, S_passive, S_gabor))
    t.start()
    print(f"Saving data in background to {filename}. Main loop continues...")

# -------------------------------
# Iterate over all ecephys sessions
# -------------------------------
for i in range(ecephys_sessions_table.shape[0]):
    ecephys_session_id = ecephys_sessions_table.index[i]
    print(f"\nProcessing session {i} with ID: {ecephys_session_id}")

    try:
        session = cache.get_ecephys_session(ecephys_session_id=ecephys_session_id)
        download_session(session, i)
    except Exception as e:
        print(f"Failed to process session {ecephys_session_id}: {e}")

