"""Load data, processes it, delete un-needed attributes, save into sample chuncks."""
import argparse
import collections
import datetime
import os
import logging
from pathlib import Path
from scipy import signal
import torch
import numpy as np
import h5py
from scipy.ndimage import binary_dilation, binary_erosion
import yaml
from tqdm import tqdm
from pympler import asizeof

from kirby.data import Data, IrregularTimeSeries, Interval
from kirby.taxonomy.taxonomy import Output, RecordingTech, Session, Task
from kirby.utils import find_files_by_extension, make_directory
from kirby.tasks.reaching import REACHING


logging.basicConfig(level=logging.INFO)


WINDOW_SIZE = 1.0
STEP_SIZE = 0.5
JITTER_PADDING = 0.25


def identify_outliers(data, threshold=6000):
    hand_acc_norm = np.linalg.norm(data.behavior.cursor_acc, axis=1)
    mask = hand_acc_norm > threshold
    structure = np.ones(100, dtype=bool)
    # Dilate the binary mask
    dilated = binary_dilation(mask, structure)
    return dilated


def extract_behavior(h5file):
    """Extract the behavior from the h5 file.

    ..note::
        Cursor position and target position are in the same frame of reference. They are both of size (sequence_len, 2).
        Finger position can be either 3d or 6d, depending on the sequence. # todo investigate more
    """
    cursor_pos = h5file["cursor_pos"][:].T
    finger_pos = h5file["finger_pos"][:].T
    target_pos = h5file["target_pos"][:].T
    timestamps = h5file["t"][:][0]

    # calculate the velocity of the cursor
    cursor_vel = np.gradient(cursor_pos, timestamps, edge_order=1, axis=0)
    cursor_acc = np.gradient(cursor_vel, timestamps, edge_order=1, axis=0)
    finger_vel = np.gradient(finger_pos, timestamps, edge_order=1, axis=0)

    # Extract two traces that capture the target and movement onsets.
    # Similar to https://www.biorxiv.org/content/10.1101/2021.11.21.469441v3.full.pdf
    cursor_vel = np.gradient(cursor_pos, timestamps, edge_order=1, axis=0)
    cursor_vel_abs = np.sqrt(cursor_vel[:, 0]**2 + cursor_vel[:, 1]**2)

    # Dirac delta whenever the target changes.
    delta_time = np.sqrt((np.diff(target_pos, axis=0) ** 2).sum(axis=1)) >= 1e-9
    delta_time = np.concatenate([delta_time, [0]])

    tics = np.where(delta_time)[0]

    thresh = .2

    # Find the maximum for each integer value of period.
    max_times = np.zeros(len(tics) - 1, dtype=int)
    reaction_times = np.zeros(len(tics) - 1, dtype=int)
    for i in range(len(tics) - 1):
        max_vel = cursor_vel_abs[tics[i]:tics[i + 1]].max()
        reaction_times[i] = np.where(cursor_vel_abs[tics[i]:tics[i + 1]] >= thresh * max_vel)[0][0]
        max_times[i] = reaction_times[i] + tics[i]

    # Transform it back to a Dirac delta.
    start_times = np.zeros_like(delta_time)
    start_times[max_times] = 1

    behavior = IrregularTimeSeries(
        timestamps=torch.tensor(timestamps),
        cursor_pos=torch.tensor(cursor_pos),
        cursor_vel=torch.tensor(cursor_vel),
        hand_vel=torch.tensor(cursor_vel)
        / 200.0,  # todo: this is used to match the other datasets
        cursor_acc=torch.tensor(cursor_acc),
        type=torch.ones(len(timestamps), dtype=torch.int64) * REACHING.RANDOM,
        target_pos=torch.tensor(target_pos),
        finger_pos=torch.tensor(finger_pos),
        finger_vel=torch.tensor(finger_vel),
        trial_onset_offset=torch.stack([torch.tensor(start_times), 
                                        torch.tensor(delta_time)], dim=1),
    )
    return behavior


def extract_lfp(h5file: h5py.File, channels: list[str] = None):
    """Extract the LFP from the h5 file."""
    logging.info("Broadband data attached. Computing LFP.")
    timestamps = h5file.get("/acquisition/timeseries/broadband/timestamps")[:].squeeze()
    broadband = h5file.get("/acquisition/timeseries/broadband/data")[:]

    # More timesteps than channels.
    assert broadband.shape[0] > broadband.shape[1]

    # A baffling sample frequency.
    fs = 24414.0625
    
    # Design and apply a low-pass filter
    nyq = 0.5 * fs # Nyquist frequency
    cutoff = 170 # remove everything above 170 Hz.
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False, output='ba')

    # Interpolation to achieve the desired sampling rate
    t_new = np.arange(timestamps[0], timestamps[-1], 1 / 500)
    lfp = np.zeros((len(t_new), broadband.shape[1]))
    for i in range(broadband.shape[1]):
        # We do this one channel at a time to save memory.
        broadband_low = signal.filtfilt(b, a, broadband[:, i], axis=0)
        lfp[:, i] = np.interp(t_new, timestamps, broadband_low)

    lfp = IrregularTimeSeries(
        timestamps=torch.tensor(t_new),
        lfp=torch.tensor(lfp),
        channels=channels,
    )

    return lfp


def load_references_2d(h5file, ref_name):
    return [[h5file[ref] for ref in ref_row] for ref_row in h5file[ref_name][:]]

def to_ascii(vector):
    return [''.join(chr(char) for char in row) for row in vector]

def extract_spikes(h5file:h5py.File, prefix:str):
    r"""This dataset has a mixture of sorted and unsorted (threshold crossings) units."""
    spikesvec = load_references_2d(h5file, "spikes")
    waveforms = load_references_2d(h5file, "wf")
    
    # This is slightly silly but we can convert channel names back to an ascii token this way.
    chan_names = to_ascii(np.array(load_references_2d(h5file, "chan_names")).squeeze())

    spikes = []
    unit_ids = []
    unit_types = []
    unit_meta = []
    unit_waveforms = []

    # The 0'th spikesvec corresponds to unsorted thresholded units, the rest are sorted.
    suffixes = ["unsorted"] + [f"sorted_{i:02}" for i in range(1, 11)]
    types = ([int(RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS)] + 
             [int(RecordingTech.UTAH_ARRAY_SPIKES)] * 10)

    # Map from common names to brodmann areas
    bas = {'s1': 3,
           'm1': 4}

    for j in range(len(spikesvec)):
        crossings = spikesvec[j]
        for i in range(len(crossings)):
            spiketimes = crossings[i][:][0]
            if spiketimes.ndim == 0:
                continue

            spikes.append(spiketimes)
            area, channel_number = chan_names[i].split(' ')

            unit_string_id = f"{prefix}/{chan_names[i]}/{suffixes[j]}"
            unit_ids.append([unit_string_id] * len(spiketimes))
            unit_types.append(np.ones_like(spiketimes, dtype=np.int64) * types[j])

            wf = np.array(waveforms[j][i][:])
            unit_meta.append(
                {'count': len(spiketimes),
                 'channel_name': chan_names[i],
                 'unit_string_id': unit_string_id,
                 'area_name': area,
                 'channel_number': channel_number,
                 'unit_number': j,
                 'ba': bas[area.lower()],
                 'type': types[j],
                 'average_waveform': wf.mean(axis=1),
                 # Based on https://zenodo.org/record/1488441
                 'waveform_sampling_rate': 24414.0625, 
                 }
            )
            unit_waveforms.append(wf)

    spikes = np.concatenate(spikes)
    unit_ids = np.concatenate(unit_ids)
    unit_types = np.concatenate(unit_types)

    # Cast to torch tensors
    unit_meta_long = {}
    for key, item in unit_meta[0].items():
        if (np.issubdtype(type(item), np.number)):
            unit_meta_long[key] = torch.tensor(np.stack([x[key] for x in unit_meta], axis=0))
        else:
            unit_meta_long[key] = np.stack([x[key] for x in unit_meta], axis=0)

    sorted = np.argsort(spikes)
    spikes = spikes[sorted]
    unit_ids = unit_ids[sorted]
    unit_types = unit_types[sorted]

    spikes = IrregularTimeSeries(
        torch.tensor(spikes),
        unit_string_id=unit_ids,
        unit_type=torch.tensor(unit_types),
    )

    units = Data(**unit_meta_long)
    return spikes, units, chan_names


def split_and_get_train_validation_test(start, end, random_state=42):
    r"""
    From the paper: We split each recording session into 10 nonoverlapping contiguous segments of equal size which
    were then categorised into three different sets: training set (8 concatenated segments), validation set (1
    segment) and testing set (1 segment)."""
    intervals = np.linspace(start, end, 11)
    start, end = intervals[:-1], intervals[1:]

    segment_ids = np.arange(10)
    rng = np.random.default_rng(random_state)
    segments_ids = rng.permutation(segment_ids)

    train_segments = list(zip(start[segments_ids[:8]], end[segments_ids[:8]]))
    valid_segments = [(start[segments_ids[8]], end[segments_ids[8]])]
    test_segments = [(start[segments_ids[9]], end[segments_ids[9]])]

    return train_segments, valid_segments, test_segments


def collect_slices(data, segments):
    slices = []
    for start, end in segments:
        slices.append(data.slice(start, end))
    return slices


if __name__ == "__main__":
    # Use argparse to extract two arguments from the command line:
    # input_dir and output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()
    raw_folder_path = args.input_dir
    processed_folder_path = args.output_dir

    make_directory(processed_folder_path, prompt_if_exists=False)
    make_directory(os.path.join(processed_folder_path, "train"))
    make_directory(os.path.join(processed_folder_path, "valid"))
    make_directory(os.path.join(processed_folder_path, "test"))

    extension = ".mat"
    configurations = collections.defaultdict(set)
    session_list = []
    # find all files with extension .mat in folder_path
    for file_path in tqdm(find_files_by_extension(raw_folder_path, extension)):
        logging.info(f"Processing file: {file_path}")
        h5file = h5py.File(file_path, "r")

        # extract behavior
        behavior = extract_behavior(h5file)
        start, end = behavior.timestamps[0], behavior.timestamps[-1]

        # extract spikes
        session_id = Path(file_path).stem
        prefix = session_id[:-3]
        assert prefix.count("_") == 1, f"Unexpected file name: {prefix}"
        spikes, units, chan_names = extract_spikes(h5file, prefix)

        # Extract LFPs
        extras = dict()
        broadband_path = Path(raw_folder_path) / "broadband" / f"{session_id}.nwb"
        # Check if the broadband data file exists.
        if broadband_path.exists():
            # Load the associated broadband data.
            broadband_file = h5py.File(broadband_path, "r")
            extras["lfp"] = extract_lfp(broadband_file, chan_names)

        data = Data(
            spikes=spikes,
            behavior=behavior,
            units=units,
            start=start,
            end=end,
            **extras
        )

        mask = identify_outliers(data)
        data.behavior.type[mask] = REACHING.OUTLIER

        # get successful trials, and keep 20% for test, 10% for validation
        (
            train_segments,
            validation_segments,
            test_segments,
        ) = split_and_get_train_validation_test(start, end)

        # collect data slices for validation and test trials
        train_slices = collect_slices(data, train_segments)
        validation_slices = collect_slices(data, validation_segments)
        test_slices = collect_slices(data, test_segments)

        # the remaining data (unstructured) is used for training
        train_buckets = []
        for segment in train_slices:
            segment.start, segment.end = 0, segment.end - segment.start
            train_buckets.extend(
                segment.bucketize(WINDOW_SIZE, STEP_SIZE, JITTER_PADDING)
            )
        # we make sure to exclude validation and test data from the training buckets
        # train_buckets = exclude_from_train(train_buckets, validation_trials + test_trials)

        # for each bucket we estimate the expected number of input tokens
        # for bucket in train_buckets:
        #     max_num_input_tokens = get_num_input_tokens(train_buckets[1], 4.0, 1.0)  # 3233, 3249, 3249, 3265, 3271, 3272, 3273, 3283, 3296, 3353
        #     batch_bucket_size = next_power_of_2(max_num_input_tokens)
        #     bucket.max_num_input_tokens = max_num_input_tokens
        #     bucket.batch_bucket_size = batch_bucket_size

        # all files are saved in their corresponding folders

        count = 0
        footprints = collections.defaultdict(list)
        for buckets, fold in [(train_buckets, "train"), (validation_slices, "valid"), (test_slices, "test")]:
            for i, sample in enumerate(buckets):
                zid = str(count).zfill(5)
                filename = os.path.splitext(os.path.basename(file_path))[0] + f"_{zid}.pt"
                path = os.path.join(processed_folder_path, fold, filename)
                torch.save(sample, path)

                footprints[fold].append({"disk": os.path.getsize(path), 
                                         "asizeof": asizeof.asizeof(sample)})

                count += 1

        session_id = os.path.splitext(os.path.basename(file_path))[0]
        footprints = dict(footprints)

        # We can safely assume that sessions occuring on the same day all have the same set of units.
        
        session_list.append(
            Session(
                subject = session_id.split("_")[0],
                date=datetime.datetime.strptime(session_id.split("_")[1], "%Y%m%d"),
                configuration=prefix,
                train_footprint=footprints["train"],
                valid_footprint=footprints["valid"],
                test_footprint=footprints["test"],
            ).__dict__()
        )

        configurations[prefix] = set(units.unit_string_id.tolist()).union(configurations[prefix])

        h5file.close()

    # Transform configurations to a list of lists, otherwise it won't serialize to yaml.
    configurations = {k: sorted(list(v)) for k, v in configurations.items()}

    # Create a description file for ease of reference.
    description = {
        "name": "odoherty_sabes",
        "description": "Reaching dataset from O'Doherty et al. (2017), data from M1 and S1.",
        "animal_model": "macaque",
        "ba": [3, 4], # Contains data from S1 and M1
        "source": "https://zenodo.org/record/583331",
        "task_type": str(Task.REACHING_CONTINUOUS),
        "inputs": [str(RecordingTech.UTAH_ARRAY_SPIKES),
                   str(RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS),
                   str(RecordingTech.UTAH_ARRAY_WAVEFORMS)],
        "outputs": [str(Output.CURSOR2D), 
                    str(Output.TARGET2D),
                    str(Output.FINGER3D), 
                    str(Output.CONTINUOUS_TRIAL_ONSET_OFFSET)],
        "folds": ["train", "valid", "test"],
        "sessions": session_list,
        "configurations": configurations,
    }

    filename = Path(processed_folder_path) / "description.yaml"
    print(f"Saving description to {filename}")

    with open(filename, "w") as f:
        yaml.dump(description, f)