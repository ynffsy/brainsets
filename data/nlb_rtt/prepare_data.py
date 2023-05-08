"""Load data, processes it, delete un-needed attributes, save into sample chuncks."""
import os
import logging
import torch
import numpy as np

from kirby.data import Data, IrregularTimeSeries, Interval
from kirby.utils import find_files_by_extension, make_directory
from kirby.tasks.reaching import REACHING

logging.basicConfig(level=logging.INFO)


WINDOW_SIZE = 1.0
STEP_SIZE = 0.5
JITTER_PADDING = 0.25


def load_file(file_path):
    # load mat file
    data = Data.load_from_nwb(file_path)
    test_flag = 'test' in file_path

    num_units = data.spikes.unit_id.max() + 1
    data.units = Data(id=torch.arange(num_units, dtype=torch.long))

    trials = data.trials

    if not test_flag:
        data.start = data.behavior.timestamps.min()
        data.end = data.behavior.timestamps.max()

        data.behavior.hand_vel = data.behavior.finger_vel / 200.
        timestamps = data.behavior.timestamps
        behavior_type = torch.ones_like(timestamps, dtype=torch.long) * REACHING.RANDOM
        test_mask = torch.zeros_like(timestamps, dtype=torch.bool)

        for i in range(len(trials)):
            test_mask[(timestamps >= (trials.start[i])) & (timestamps < (trials.end[i]))] = True

        data.behavior.type = behavior_type
        data.behavior.test_mask = test_mask
    return data


def split_and_get_train_validation(trials):
    train_ids = np.where(trials.split == 'train')[0]
    valid_ids = np.where(trials.split == 'val')[0]

    train_trials = [trials[i] for i in train_ids]
    valid_trials = [trials[i] for i in valid_ids]

    return train_trials, valid_trials


def collect_slices(data, trials, min_duration=WINDOW_SIZE):
    slices = []
    for trial in trials:
        start, end = trial['start'] - 1.0, trial['end'] + 1.0
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
    make_directory(os.path.join(processed_folder_path, 'finetune'))
    
    extension = ".nwb"

    session_list = []
    # find all files with extension .nwb in folder_path
    for file_path in find_files_by_extension(raw_folder_path, extension):
        test_flag = 'test' in file_path

        if test_flag: 
            continue

        logging.info(f"Processing file: {file_path}")
        data = load_file(file_path)

        # collect data slices for validation and test trials
        if not test_flag:
            train_trials, validation_trials = split_and_get_train_validation(data.trials)
            
            train_slices = collect_slices(data, train_trials)
            validation_slices = collect_slices(data, validation_trials)
        
            # the remaining data (unstructured) is used for training
            train_buckets = list(data.bucketize(WINDOW_SIZE, STEP_SIZE, JITTER_PADDING))
            # we make sure to exclude validation and test data from the training buckets
            train_buckets = exclude_from_train(train_buckets, validation_trials)

            for i, sample in enumerate(train_buckets):
                zid = str(i).zfill(5)
                filename = os.path.splitext(os.path.basename(file_path))[0] + f'_{zid}.pt'
                path = os.path.join(processed_folder_path, 'train', filename)
                torch.save(sample, path)

            count = len(train_buckets)
            for i, sample in enumerate(train_slices):
                zid = str(i + count).zfill(5)
                filename = os.path.splitext(os.path.basename(file_path))[0] + f'_{zid}.pt'
                path = os.path.join(processed_folder_path, 'finetune', filename)
                torch.save(sample, path)

            count += len(train_slices)
            for i, sample in enumerate(validation_slices):
                zid = str(i + count).zfill(5)
                filename = os.path.splitext(os.path.basename(file_path))[0] + f'_{zid}.pt'
                path = os.path.join(processed_folder_path, 'valid', filename)
                torch.save(sample, path)
        
        else:
            test_slices = collect_slices(data, data.trials)

            for i, sample in enumerate(test_slices):
                zid = str(i).zfill(5)
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
