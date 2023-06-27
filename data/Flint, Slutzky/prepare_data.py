from scipy.io import loadmat
import numpy as np

import os
import logging
import torch
import torch
from sklearn.model_selection import train_test_split
import numpy as np


from kirby.utils import find_files_by_extension, make_directory
from kirby.data import Data, IrregularTimeSeries, Interval
from kirby.tasks.reaching import REACHING


logging.basicConfig(level=logging.INFO)


WINDOW_SIZE = 1.0
STEP_SIZE = 0.5
JITTER_PADDING = 0.25


def process_mat(mat):
    keys = mat['Subject'][0][0][0].dtype.names
    values = mat['Subject'][0][0][0]

    data_list = []

    for trial_id in range(len(values['Time'])):
        timestamps = values['Time'][trial_id][0][:, 0]
        hand_vel = values['HandVel'][trial_id][0][:, :2]
        start = timestamps.min()

        behavior = IrregularTimeSeries(
            timestamps=torch.tensor(timestamps) - start,
            hand_vel=torch.tensor(hand_vel) * 2.5,
            type=torch.tensor(np.ones_like(timestamps) * REACHING.RANDOM,dtype=torch.long),
            )

        neurons = values['Neuron'][trial_id][0]

        spikes = []
        unit_ids = []
        unit_meta = []
        for i in range(len(neurons)):
            unit_meta.append((i,))

            spiketimes = neurons[i][0][0]
            if len(spiketimes) == 0:
                continue
            spiketimes = spiketimes[:, 0] - start
            
            spikes.append(spiketimes)
            unit_ids.append(np.ones_like(spiketimes, dtype=np.int64) * i)

        spikes = np.concatenate(spikes)
        unit_ids = np.concatenate(unit_ids)
        
        unit_meta = dict(zip(('id',) , zip(*unit_meta)))
        for key in unit_meta:
            if np.issubdtype(type(unit_meta[key][0]), np.number):
                unit_meta[key] = torch.tensor(unit_meta[key])

        sorted = np.argsort(spikes)
        spikes = spikes[sorted]
        unit_ids = unit_ids[sorted]

        spikes = IrregularTimeSeries(torch.tensor(spikes), unit_id=torch.tensor(unit_ids))
        units = Data(**unit_meta)

        data = Data(
            spikes=spikes,
            behavior=behavior,
            units=units,
            start=0.,
            end=behavior.timestamps.max() - start,
            )
        data_list.append(data)
    return data_list


def split_and_get_train_validation_test(trials, test_size=0.2, valid_size=0.1, random_state=42):
    assert 0 < valid_size < 1, "valid_size must be positive, got {}".format(valid_size)
    assert 0 < test_size < 1, "test_size must be positive, got {}".format(test_size)

    num_trials = len(trials)
    train_size = 1. - test_size - valid_size
    assert 0 < train_size < 1, "train_size must be positive, got {}".format(train_size)
    
    train_valid_ids, test_ids = train_test_split(np.arange(num_trials), test_size=test_size, random_state=random_state)
    train_ids, valid_ids = train_test_split(train_valid_ids, test_size=valid_size/(train_size+valid_size), random_state=random_state)
    
    train_trials = [trials[i] for i in train_ids]
    valid_trials = [trials[i] for i in valid_ids]
    test_trials = [trials[i] for i in test_ids]

    return train_trials, valid_trials, test_trials


if __name__ == "__main__":
    raw_folder_path = "./raw"
    processed_folder_path = "./processed"
    make_directory(processed_folder_path, prompt_if_exists=True)
    make_directory(os.path.join(processed_folder_path, 'train'))
    make_directory(os.path.join(processed_folder_path, 'finetune'))
    make_directory(os.path.join(processed_folder_path, 'valid'))
    make_directory(os.path.join(processed_folder_path, 'test'))

    extension = ".mat"
    session_list = []
    for file_path in find_files_by_extension(raw_folder_path, extension):
        logging.info(f"Processing file: {file_path}")
        mat = loadmat(file_path)
        data_list = process_mat(mat)

        train_trials, validation_trials, test_trials = split_and_get_train_validation_test(np.arange(len(data_list)), test_size=0.2, valid_size=0.1, 
                                                                       random_state=42)

        # collect data slices for validation and test trials
        train_slices = [data_list[i] for i in train_trials]
        validation_slices = [data_list[i] for i in validation_trials]
        test_slices = [data_list[i] for i in test_trials]

        train_buckets = []
        for slice in train_slices:
            train_buckets.extend(list(slice.bucketize(WINDOW_SIZE, STEP_SIZE, JITTER_PADDING)))

        # all files are saved in their corresponding folders
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

        count += len(validation_slices)
        for i, sample in enumerate(test_slices):
            zid = str(i + count).zfill(5)
            filename = os.path.splitext(os.path.basename(file_path))[0] + f'_{zid}.pt'
            path = os.path.join(processed_folder_path, 'test', filename)
            torch.save(sample, path)

        session_id = os.path.splitext(os.path.basename(file_path))[0]
        num_units = len(data_list[0].units.id)
        session_list.append((session_id, num_units))

    # save session_list as txt
    info_path = os.path.join('.', 'all.txt')
    with open(info_path, 'w') as f:
        for session_id, num_units in session_list:
            f.write(f'{session_id} {num_units}\n')
