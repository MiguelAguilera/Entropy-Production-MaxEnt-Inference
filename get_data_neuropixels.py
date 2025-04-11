import numpy as np
from pathlib import Path
import argparse
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache


# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Estimate EP for the spin model with varying beta values.")

parser.add_argument("--BASE_DIR", type=str, default="~/Neuropixels",
                    help="Base directory to store the data (default: '~/NeuroPixels').")


args = parser.parse_args()

# Output directory and cache loading
output_dir = Path(args.BASE_DIR)
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=output_dir)
cache.load_latest_manifest()

# Load tables from the cache
units_table = cache.get_unit_table()
channels_table = cache.get_channel_table()
probes_table = cache.get_probe_table()
behavior_sessions_table = cache.get_behavior_session_table()
ecephys_sessions_table = cache.get_ecephys_session_table()

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
    S = np.zeros((N, T), dtype=bool)

    for u, unit_id in enumerate(unit_ids):
        spike_times_discretized = np.floor((np.array(spike_times[unit_id]) - minT) / bin_size).astype(int)
        spike_times_discretized = spike_times_discretized[
            (spike_times_discretized >= 0) & (spike_times_discretized < T)
        ]
        S[u, spike_times_discretized] = True

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
    S_active = S[:, active_mask]
    S_passive = S[:, passive_mask]
    S_gabor = S[:, gabor_mask]

    # Save processed data
    filename = output_dir / 'parallel_update' / f'data_binsize_{bin_size}_session_{ind}.npz'
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filename, S_active=S_active, S_passive=S_passive, S_gabor=S_gabor, areas=areas)

    print(f"Saved processed data for session {ind} to {filename}")

# Iterate over all ecephys sessions
for i in range(ecephys_sessions_table.shape[0]):
    ecephys_session_id = ecephys_sessions_table.index[i]
    print(f"\nProcessing session {i} with ID: {ecephys_session_id}")

    try:
        session = cache.get_ecephys_session(ecephys_session_id=ecephys_session_id)
        download_session(session, i)
    except Exception as e:
        print(f"Failed to process session {ecephys_session_id}: {e}")

