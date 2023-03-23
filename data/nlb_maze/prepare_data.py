"""Load data, processes it, delete un-needed attributes, save into sample chuncks."""
import os
import logging
import torch

from src.data import Data, IrregularTimeSeries, Interval
from src.utils import find_files_by_extension, make_directory

logging.basicConfig(level=logging.INFO)

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
        data = Data.load_from_nwb(file_path)

        test_flag = 'test' in file_path

        # remove all attributes that are not needed
        data_lite = Data()
        # keep spikes and hand_vel
        data_lite.spikes = data.spikes
        # keep trial
        data_lite.trials = Interval(start=data.trials.move_onset_time - 0.15,
                                    end  =data.trials.move_onset_time + 0.55)

        if not test_flag:
            data_lite.behavior = IrregularTimeSeries(data.behavior.timestamps, hand_vel=data.behavior.hand_vel)

        # split data according to trial_start
        data_iter = data_lite.slice_along('trials', 'start', 0.7)

        if not test_flag:
            # use the default train/val split from NLB
            train_mask = data.trials.split == "train"

        session_id = os.path.splitext(os.path.basename(file_path))[0]
        num_units = data.spikes.unit_id.max() + 1
        session_list.append((session_id, num_units))
        # iterate over all samples, and save each in a .pt file
        for i, sample in enumerate(data_iter):
            zid = str(i).zfill(5)
            if not test_flag:
                folder = 'train' if train_mask[i] else 'valid'
            else:
                folder = 'test'
            filename = session_id + f'_{zid}.pt'
            path = os.path.join(processed_folder_path, folder, filename)
            torch.save(sample, path)

    # save session_list as txt
    info_path = os.path.join(processed_folder_path, 'info.txt')
    with open(info_path, 'w') as f:
        for session_id, num_units in session_list:
            f.write(f'{session_id} {num_units}\n')
