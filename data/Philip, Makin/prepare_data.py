"""Load data, processes it, delete un-needed attributes, save into sample chuncks."""
import os
import logging
import torch
import numpy as np
import h5py

from kirby.data import Data, IrregularTimeSeries, Interval
from kirby.utils import find_files_by_extension, make_directory
from kirby.tasks.reaching import REACHING


logging.basicConfig(level=logging.INFO)


WINDOW_SIZE = 1.0
STEP_SIZE = 0.5
JITTER_PADDING = 0.25


def extract_behavior(h5file):
    """Extract the behavior from the h5 file.
    
    ..note::
        Cursor position and target position are in the same frame of reference. They are both of size (sequence_len, 2).
        Finger position can be either 3d or 6d, depending on the sequence. # todo investigate more
    """
    cursor_pos = h5file['cursor_pos'][:].T
    finger_pos = h5file['finger_pos'][:].T
    target_pos = h5file['target_pos'][:].T
    timestamps = h5file['t'][:][0]

    # calculate the velocity of the cursor
    cursor_vel = np.gradient(cursor_pos, timestamps, edge_order=1, axis=0)
    finger_vel = np.gradient(finger_pos, timestamps, edge_order=1, axis=0)

    behavior = IrregularTimeSeries(
        timestamps=torch.tensor(timestamps),
        cursor_pos=torch.tensor(cursor_pos),
        cursor_vel=torch.tensor(cursor_vel),
        hand_vel=torch.tensor(cursor_vel) / 200.,  # todo: this is used to match the other datasets
        behavior_type=torch.ones(len(timestamps), dtype=torch.int64) * REACHING.RANDOM,
        target_pos=torch.tensor(target_pos),
        finger_pos=torch.tensor(finger_pos),
        finger_vel=torch.tensor(finger_vel),
    )
    return behavior


def load_references_2d(h5file, ref_name):
    return np.array([[h5file[ref] for ref in ref_row] for ref_row in h5file[ref_name][:]])

def extract_spikes(h5file):
    r"""This dataset has a mixture of sorted and unsorted (threshold crossings) units."""
    spikesvec = load_references_2d(h5file, 'spikes')

    spikes = []
    unit_ids = []
    unit_types = []
    unit_meta = []
    unit_count = 0
    # hash units first
    hash_units = spikesvec[0]
    for i in range(len(hash_units)):
        spiketimes = hash_units[i][:][0]
        if spiketimes.ndim == 0:
            continue
        spikes.append(spiketimes)
        unit_ids.append(np.ones_like(spiketimes, dtype=np.int64) * unit_count)
        unit_types.append(np.ones_like(spiketimes, dtype=np.int64) * 1)
        unit_meta.append((unit_count, i))
        unit_count += 1

    # then non-hash units
    sorted_units_dim = spikesvec.shape[0] - 1
    for i in range(1, sorted_units_dim + 1):
        sorted_units = spikesvec[i]
        for j in range(len(sorted_units)):
            if sorted_units[j].ndim == 2:
                spiketimes = sorted_units[j][:][0]
                spikes.append(spiketimes)
                unit_ids.append(np.ones_like(spiketimes, dtype=np.int64) * unit_count)
                unit_types.append(np.ones_like(spiketimes, dtype=np.int64) * 0)
                unit_meta.append((unit_count, i))
                unit_count += 1

    spikes = np.concatenate(spikes)
    unit_ids = np.concatenate(unit_ids)
    unit_types = np.concatenate(unit_types)

    unit_meta = dict(zip(('id', 'channel') , zip(*unit_meta)))
    for key in unit_meta:
        if np.issubdtype(type(unit_meta[key][0]), np.number):
            unit_meta[key] = torch.tensor(unit_meta[key])

    sorted = np.argsort(spikes)
    spikes = spikes[sorted]
    unit_ids = unit_ids[sorted]
    unit_types = unit_types[sorted]

    spikes = IrregularTimeSeries(
        torch.tensor(spikes), 
        unit_id=torch.tensor(unit_ids),
        unit_type=torch.tensor(unit_types)
    )

    units = Data(**unit_meta)
    return spikes, units


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
    raw_folder_path = "./raw"
    processed_folder_path = "./processed"
    make_directory(processed_folder_path, prompt_if_exists=True)
    make_directory(os.path.join(processed_folder_path, 'train'))
    make_directory(os.path.join(processed_folder_path, 'valid'))
    make_directory(os.path.join(processed_folder_path, 'test'))

    extension = ".mat"
    session_list = []
    # find all files with extension .mat in folder_path
    for file_path in find_files_by_extension(raw_folder_path, extension):
        logging.info(f"Processing file: {file_path}")
        h5file = h5py.File(file_path, 'r')

        # extract behavior
        behavior = extract_behavior(h5file)
        start, end = behavior.timestamps[0], behavior.timestamps[-1]

        # extract spikes
        spikes, units = extract_spikes(h5file)

        data = Data(spikes=spikes, behavior=behavior, units=units, start=start, end=end,)

        # get successful trials, and keep 20% for test, 10% for validation
        train_segments, validation_segments, test_segments = split_and_get_train_validation_test(start, end)

        # collect data slices for validation and test trials
        train_slices = collect_slices(data, train_segments)
        validation_slices = collect_slices(data, validation_segments)
        test_slices = collect_slices(data, test_segments)

        # the remaining data (unstructured) is used for training
        train_buckets = []
        for segment in train_slices:
            segment.start, segment.end = 0, segment.end - segment.start
            train_buckets.extend(segment.bucketize(WINDOW_SIZE, STEP_SIZE, JITTER_PADDING))
        # we make sure to exclude validation and test data from the training buckets
        # train_buckets = exclude_from_train(train_buckets, validation_trials + test_trials)

        # for each bucket we estimate the expected number of input tokens
        # for bucket in train_buckets:
        #     max_num_input_tokens = get_num_input_tokens(train_buckets[1], 4.0, 1.0)  # 3233, 3249, 3249, 3265, 3271, 3272, 3273, 3283, 3296, 3353
        #     batch_bucket_size = next_power_of_2(max_num_input_tokens)
        #     bucket.max_num_input_tokens = max_num_input_tokens
        #     bucket.batch_bucket_size = batch_bucket_size

        # all files are saved in their corresponding folders
        for i, sample in enumerate(train_buckets):
            zid = str(i).zfill(5)
            filename = os.path.splitext(os.path.basename(file_path))[0] + f'_{zid}.pt'
            path = os.path.join(processed_folder_path, 'train', filename)
            torch.save(sample, path)

        count = len(train_buckets)
        for i, sample in enumerate(validation_slices):
            zid = str(i + count).zfill(5)
            filename = os.path.splitext(os.path.basename(file_path))[0] + f'_{zid}.pt'
            path = os.path.join(processed_folder_path, 'valid', filename)
            torch.save(sample, path)

        count += len(validation_slices)
        for i, sample in enumerate(test_slices):
            zid = str(i + count).zfill(5)
            filename = os.path.splitext(os.path.basename(file_path))[0] + f'_{zid}.pt'
            path = os.path.join(processed_folder_path, 'test', filename)
            torch.save(sample, path)

        session_id = os.path.splitext(os.path.basename(file_path))[0]
        num_units = data.spikes.unit_id.max() + 1
        session_list.append((session_id, num_units))

        h5file.close()

    # save session_list as txt
    info_path = os.path.join('.', 'all.txt')
    with open(info_path, 'w') as f:
        for session_id, num_units in session_list:
            f.write(f'{session_id} {num_units}\n')
