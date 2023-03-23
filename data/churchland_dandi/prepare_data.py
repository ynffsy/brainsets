"""Load data, processes it, delete un-needed attributes, save into sample chuncks."""
import os
import logging
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data import Data, IrregularTimeSeries, Interval
from src.utils import find_files_by_extension, make_directory

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    raw_folder_path = "./raw"
    processed_folder_path = "./processed"
    make_directory(processed_folder_path, prompt_if_exists=True)
    make_directory(os.path.join(processed_folder_path, 'train'))
    make_directory(os.path.join(processed_folder_path, 'valid'))

    extension = ".nwb"
    session_list = []
    # find all files with extension .nwb in folder_path
    for file_path in find_files_by_extension(raw_folder_path, extension):
        logging.info(f"Processing file: {file_path}")
        data = Data.load_from_nwb(file_path)

        # remove all attributes that are not needed
        data_lite = Data()
        # keep spikes
        data_lite.spikes = data.spikes
        # compute hand_vel from hand_pos
        hand_pos = data.behavior_0.Position_Hand.numpy()
        timestamps = data.behavior_0.timestamps.numpy()
        hand_vel = torch.DoubleTensor(np.gradient(hand_pos, timestamps, edge_order=1, axis=0))
        data_lite.behavior = IrregularTimeSeries(data.behavior_0.timestamps, hand_vel=hand_vel)
        # keep trial
        data_lite.trials = Interval(start=data.trials.move_begins_time - 0.15,
                                    end  =data.trials.move_begins_time + 0.55)

        # split data according to trial_start
        data_iter = data_lite.slice_along('trials', 'start', 0.7)

        # randomly assign to train/valid
        num_trials = len(data.trials)
        random_seed= 42
        train_size = 0.75
        train_ids, valid_ids = train_test_split(np.arange(num_trials), train_size=train_size, random_state=random_seed)
        train_mask = np.zeros(num_trials, dtype=bool)
        train_mask[train_ids] = True

        session_id = os.path.splitext(os.path.basename(file_path))[0]
        num_units = data.spikes.unit_id.max() + 1
        session_list.append((session_id, num_units))
        # iterate over all samples, and save each in a .pt file

        for i, sample in enumerate(data_iter):
            num_timepoints = sample.behavior.hand_vel.shape[0]
            if num_timepoints < 600:
                continue
            zid = str(i).zfill(5)
            folder = 'train' if train_mask[i] else 'valid'
            filename = os.path.splitext(os.path.basename(file_path))[0] + f'_{zid}.pt'
            path = os.path.join(processed_folder_path, folder, filename)
            torch.save(sample, path)

    # save session_list as txt
    info_path = os.path.join(processed_folder_path, 'info.txt')
    with open(info_path, 'w') as f:
        for session_id, num_units in session_list:
            f.write(f'{session_id} {num_units}\n')
