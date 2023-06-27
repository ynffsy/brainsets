"""Load data, processes it, delete un-needed attributes, save into sample chuncks."""
import os
import logging
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pynwb import NWBHDF5IO

from kirby.utils import find_files_by_extension, make_directory
from kirby.data import Data, IrregularTimeSeries, Interval
from kirby.tasks.reaching import REACHING

logging.basicConfig(level=logging.INFO)


WINDOW_SIZE = 1.0
STEP_SIZE = 0.5
JITTER_PADDING = 0.25


def load_file(file_path):
    with NWBHDF5IO(file_path, 'r') as io:
        nwbfile = io.read()

        behavior = extract_behavior(nwbfile)
        spikes, units = extract_spikes(nwbfile)
        trials, behavior = extract_trials(nwbfile, behavior)


    start, end = behavior.timestamps[0], behavior.timestamps[-1]
    data = Data(
        start=start,
        end=end,
        trials=trials,
        behavior=behavior,
        spikes=spikes,
        units=units,
    )
    return data


def extract_behavior(nwbfile):
    """Extract the behavior from the h5 file.
    
    ..note::
        Cursor position and target position are in the same frame of reference. They are both of size (sequence_len, 2).
        Finger position can be either 3d or 6d, depending on the sequence. # todo investigate more
    """
    cursor_pos = nwbfile.processing['behavior']['Position']['Cursor'].data[:]
    timestamps = nwbfile.processing['behavior']['Position']['Cursor'].timestamps[:]
    hand_pos = nwbfile.processing['behavior']['Position']['Hand'].data[:]
    # hand_timestamps = nwbfile.processing['behavior']['Position']['Hand'].timestamps[:]
    eye_pos = nwbfile.processing['behavior']['Position']['Eye'].data[:]
    # eye_timestamps = nwbfile.processing['behavior']['Position']['Eye'].timestamps[:]

    # calculate the velocity of the cursor
    cursor_vel = np.gradient(cursor_pos, timestamps, edge_order=1, axis=0)
    hand_vel = np.gradient(hand_pos, timestamps, edge_order=1, axis=0)

    behavior = IrregularTimeSeries(
        timestamps=torch.tensor(timestamps),
        cursor_pos=torch.tensor(cursor_pos),
        cursor_vel=torch.tensor(cursor_vel),
        hand_pos=torch.tensor(hand_pos),
        hand_vel=torch.tensor(hand_vel) / 4000.,
        eye_pos=torch.tensor(eye_pos),
    )
    return behavior


def extract_spikes(nwbfile):
    r"""This dataset has a mixture of sorted and unsorted (threshold crossings) units."""
    spikes_per_unit = nwbfile.units.spike_times_index[:]
    unit_table = nwbfile.units.electrodes

    spikes = []
    unit_ids = []
    unit_types = []
    unit_meta = []
    unit_count = 0

    # all these units are obtained using threshold crossings
    for i in range(len(spikes_per_unit)):
        spikes.append(spikes_per_unit[i])
        unit_ids.append(np.ones_like(spikes_per_unit[i], dtype=np.int64) * unit_count)
        unit_types.append(np.ones_like(spikes_per_unit[i], dtype=np.int64) * 1)
        unit_meta.append((unit_count, unit_table[i]['location'].item()))
        unit_count += 1

    spikes = np.concatenate(spikes)
    unit_ids = np.concatenate(unit_ids)
    unit_types = np.concatenate(unit_types)

    unit_meta = dict(zip(('id', 'region') , zip(*unit_meta)))
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


def extract_trials(nwbfile, behavior):
    trial_table = nwbfile.trials.to_dataframe()
    
    trials = Interval(
        start=torch.tensor(trial_table.start_time.values),
        end=torch.tensor(trial_table.reach_time.values),
        is_successful=torch.tensor(trial_table.is_successful.values),
        task_id=torch.tensor(trial_table.task_id.values),
        target_on_time=torch.tensor(trial_table.target_shown_time.values),
        go_cue_time=torch.tensor(trial_table.go_cue_time.values),
    )

    timestamps = behavior.timestamps
    behavior_type = torch.ones_like(timestamps, dtype=torch.long) * REACHING.RANDOM
    # go_cue_event = []
    # target_acquired_event = []
    # target_direction_timestamps = []
    # target_direction = []

    for i in range(len(trials)):
        behavior_type[(timestamps >= trials.start[i]) & (timestamps < trials.go_cue_time[i])] = REACHING.HOLD

        #     if not np.isnan(trials.target_dir[i]):
        #         target_direction_timestamps.append((trials.go_cue_time[i] + trials.end_time[i]) * 0.5)
        #         target_direction.append(trials.target_dir[i])

        # if not np.isnan(trials.go_cue_time[i]):
        #     go_cue_event.append(trials.go_cue_time[i])
        # if not np.isnan(trials.end_time[i]):
        #     target_acquired_event.append(trials.end_time[i])
    
    behavior_type[torch.norm(behavior.hand_vel, dim=1) > 5.] = REACHING.OUTLIER

    behavior.type = behavior_type
    # target_direction = IrregularTimeSeries(timestamps=torch.tensor(target_direction_timestamps), 
    #                                        direction=torch.tensor(target_direction))
    # go_cue = IrregularTimeSeries(timestamps=torch.tensor(go_cue_event))
    # target_acquired = IrregularTimeSeries(timestamps=torch.tensor(target_acquired_event))

    return trials, behavior


def check_trial_validity(trial):
    # check if the trial was successful
    return trial['is_successful'] == 1.



def split_and_get_validation_test(trials, test_size=0.2, valid_size=0.1, random_state=42):
    assert 0 < valid_size < 1, "valid_size must be positive, got {}".format(valid_size)
    assert 0 < test_size < 1, "test_size must be positive, got {}".format(test_size)

    num_trials = len(trials)
    train_size = 1. - test_size - valid_size
    assert 0 < train_size < 1, "train_size must be positive, got {}".format(train_size)
    
    train_valid_ids, test_ids = train_test_split(np.arange(num_trials), test_size=test_size, random_state=random_state)
    train_ids, valid_ids = train_test_split(train_valid_ids, test_size=valid_size/(train_size+valid_size), random_state=random_state)
    
    valid_trials = [trials[i] for i in valid_ids]
    test_trials = [trials[i] for i in test_ids]

    return valid_trials, test_trials


def collect_slices(data, trials, min_duration=WINDOW_SIZE):
    slices = []
    for trial in trials:
        start, end = trial['start'], trial['end']
        if end - start <= min_duration:
            start = start - (min_duration - (end - start)) / 2
            end = start + min_duration
        slices.append(data.slice(start, end))
    return slices


def exclude_from_train(buckets, exclude_trials):
    out = []
    for i in range(len(buckets)):
        exclude = False
        for trial in exclude_trials:
            start, end = trial['start'], trial['end']
            bucket_start, bucket_end = buckets[i].start, buckets[i].end
            if start <= bucket_end and end >= bucket_start:
                exclude = True
                break
        if not exclude:
            out.append(buckets[i])
    return out



if __name__ == "__main__":
    raw_folder_path = "./raw"
    processed_folder_path = "./processed"
    make_directory(processed_folder_path, prompt_if_exists=True)
    make_directory(os.path.join(processed_folder_path, 'train'))
    make_directory(os.path.join(processed_folder_path, 'valid'))
    make_directory(os.path.join(processed_folder_path, 'test'))

    extension = ".nwb"
    session_list = []
    # find all files with extension .nwb in folder_path
    for file_path in find_files_by_extension(raw_folder_path, extension):
        logging.info(f"Processing file: {file_path}")
        
        # remove all attributes that are not needed
        data = load_file(file_path)
        
        # get successful trials, and keep 20% for test, 10% for validation
        valid_trials = list(filter(check_trial_validity, data.trials))
        validation_trials, test_trials = split_and_get_validation_test(valid_trials, test_size=0.2, valid_size=0.1, 
                                                                       random_state=42)

        # collect data slices for validation and test trials
        validation_slices = collect_slices(data, validation_trials)
        test_slices = collect_slices(data, test_trials)

        # the remaining data (unstructured) is used for training
        train_buckets = list(data.bucketize(WINDOW_SIZE, STEP_SIZE, JITTER_PADDING))
        # we make sure to exclude validation and test data from the training buckets
        train_buckets = exclude_from_train(train_buckets, validation_trials + test_trials)
 
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

    # save session_list as txt
    info_path = os.path.join('.', 'all.txt')
    with open(info_path, 'w') as f:
        for session_id, num_units in session_list:
            f.write(f'{session_id} {num_units}\n')
