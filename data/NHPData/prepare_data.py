"""Load data, processes it, delete un-needed attributes, save into sample chuncks."""
import os
import logging
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import scipy

from src.data import Data, IrregularTimeSeries, Interval
from src.utils import find_files_by_extension, make_directory

logging.basicConfig(level=logging.INFO)



def reformat_spikes(units):
    spikes = []
    unit_ids = []
    unit_meta = []
    for i in range(len(units)):
        region = units[i][1][0]
        electrode = units[i][4][0]
        bank = units[i][2][0]
        pin = units[i][3][0]
        try:
            row, col = units[i][6][0, 0], units[i][5][0, 0]
        except:
            row, col = torch.nan, torch.nan
        spiketimes = units[i][7][0][0][2][0, 0][:, 0]
        waveforms = units[i][7][0][0][2][0, 1][:, 0]

        unit_meta.append((i, region, electrode, bank, pin, row, col))
        spikes.append(spiketimes)
        unit_ids.append(np.ones_like(spiketimes, dtype=np.int64) * i)
    
    spikes = np.concatenate(spikes)
    unit_ids = np.concatenate(unit_ids)
    
    unit_meta = dict(zip(('id', 'region', 'electrode', 'bank', 'pin', 'row', 'col') , zip(*unit_meta)))
    for key in unit_meta:
        if np.issubdtype(type(unit_meta[key][0]), np.number):
            unit_meta[key] = torch.tensor(unit_meta[key])

    sorted = np.argsort(spikes)
    spikes = spikes[sorted]
    unit_ids = unit_ids[sorted]
    return spikes, unit_ids, unit_meta


if __name__ == "__main__":
    raw_folder_path = "./raw/ReachingData"
    processed_folder_path = "./processed"
    make_directory(processed_folder_path, prompt_if_exists=True)
    make_directory(os.path.join(processed_folder_path, 'train'))
    make_directory(os.path.join(processed_folder_path, 'valid'))
    make_directory(os.path.join(processed_folder_path, 'test'))

    extension = ".mat"
    session_list = []
    # find all files with extension .nwb in folder_path
    for file_path in find_files_by_extension(raw_folder_path, extension):
        if not "BL" in file_path:
            continue
        logging.info(f"Processing file: {file_path}")
        mat_dict = scipy.io.loadmat(file_path)

        # behavior
        t, x, y, vx, vy, ax, ay = mat_dict['data'][0][0][2][0, 0]
        behavior = IrregularTimeSeries(
            timestamps=torch.tensor(t[:, 0]),
            hand_pos=torch.tensor(np.concatenate([x, y], axis=1)),
            hand_vel=torch.tensor(np.concatenate([vx, vy], axis=1)),
            hand_acc=torch.tensor(np.concatenate([ax, ay], axis=1)), )

        # trial table
        keys = mat_dict['data'][0][0][1][0].dtype.names
        values = mat_dict['data'][0][0][1][0][0]
        
        if "CO" in file_path:
            # the end is when the target for the next trial is shown
            target_next_trial = np.append(values['tgtOnTime'][1:, 0], values['endTime'][-1, 0]+1.)
            default_value = values['endTime'][:, 0] + 1.
            nan_mask = np.isnan(target_next_trial)
            target_next_trial[nan_mask] = default_value[nan_mask]

            trials = Interval(start=torch.tensor(values['tgtOnTime'][:, 0]),
                              end=torch.tensor(target_next_trial),
                              # other events
                              start_time=torch.tensor(values['startTime'][:, 0]),
                              target_on_time=torch.tensor(values['tgtOnTime'][:, 0]),
                              go_cue_time=torch.tensor(values['goCueTime'][:, 0]),
                              end_time=torch.tensor(values['endTime'][:, 0]),
                              target_id=torch.tensor(values['tgtID'][:, 0]),
                              target_dir=torch.tensor(values['tgtDir'][:, 0]),
                              result=values['result'],
                              )

            timestamps = behavior.timestamps
            behavior_type = torch.zeros_like(timestamps, dtype=torch.long)
            for i in range(len(trials)):
                behavior_type[(timestamps >= trials.target_on_time[i]) & (timestamps < trials.go_cue_time[i])] = 1
                behavior_type[(timestamps >= trials.go_cue_time[i]) & (timestamps < trials.end_time[i])] = 2
                behavior_type[(timestamps >= trials.end_time[i]) & (timestamps < trials.end[i])] = 3
            behavior.behavior_type = behavior_type

        elif "RT" in file_path:
            trials = Interval(start=torch.tensor(values['startTime'][:, 0]),
                              end=torch.tensor(values['endTime'][:, 0]),
                              # other events
                              start_time=torch.tensor(values['startTime'][:, 0]),
                              go_cue_time=torch.tensor(values['goCueTime'][:, 0]),
                              end_time=torch.tensor(values['endTime'][:, 0]),
                              result=values['result'],
                              num_attempts=torch.tensor(values['numAttempted'][:, 0]),
                              )
            
            timestamps = behavior.timestamps
            behavior_type = torch.zeros_like(timestamps, dtype=torch.long)
            for i in range(len(trials)):
                behavior_type[(timestamps >= trials.start_time[i]) & (timestamps < trials.go_cue_time[i])] = 1
                behavior_type[(timestamps >= trials.go_cue_time[i]) & (timestamps < trials.end_time[i])] = 4
            behavior.behavior_type = behavior_type

        else:
            raise ValueError("Unknown session type")

        # unit activity
        units = mat_dict['data'][0][0][3][0]
        spikes, unit_id, unit_meta = reformat_spikes(units)
        spikes = IrregularTimeSeries(torch.tensor(spikes), unit_id=torch.tensor(unit_id))
        units = Data(**unit_meta)
        data = Data(spikes=spikes, behavior=behavior, trials=trials, units=units)

        # split data according to trial_start
        data_iter = data.slice_along('trials', 'start', 'end')

        # randomly assign to train/valid
        num_trials = len(data.trials)
        random_seed= 42
        train_size = 0.7
        valid_size = 0.1
        test_size = 0.2
        train_valid_ids, test_ids = train_test_split(np.arange(num_trials), test_size=test_size, random_state=random_seed)
        train_ids, valid_ids = train_test_split(train_valid_ids, test_size=valid_size/(train_size+valid_size), random_state=random_seed)
        train_mask = np.zeros(num_trials, dtype=bool)
        valid_mask = np.zeros(num_trials, dtype=bool)
        train_mask[train_ids] = True
        valid_mask[valid_ids] = True

        session_id = os.path.splitext(os.path.basename(file_path))[0]
        num_units = data.spikes.unit_id.max() + 1
        session_list.append((session_id, num_units))
        # iterate over all samples, and save each in a .pt file

        skip_count = 0
        trial_length_min_max = [10, 0]
        for i, sample in enumerate(data_iter):
            num_timepoints = sample.behavior.hand_vel.shape[0]
            if sample.trials['result'] != 'R':
                skip_count += 1
                continue

            if sample.spikes.timestamps.shape[0] <= 50:
                print('Skipped because of too few spikes')
                skip_count += 1
                continue

            if (sample.trials['end'] - sample.trials['start']) < 1.5:
                print('Skipped because of too short trial')
                skip_count += 1
                continue

            trial_length = sample.trials['end'] - sample.trials['start']
            if trial_length > 6:
                print('Skipped trial because it was longer than 6s')
                continue

            if "CO" in file_path:
                if sample.trials['target_id'].isnan():
                    print('Skipped because of nan target id', sample.trials['target_id'])
                    skip_count += 1
                    continue
            elif "RT" in file_path:
                if sample.trials['num_attempts'] != 4:
                    skip_count += 1
                    continue

            trial_length_min_max = [min(trial_length_min_max[0], trial_length), max(trial_length_min_max[1], trial_length)]
            # sample.behavior.clip(start=sample.trials['goCueTime'], end=sample.trials['end'])

            zid = str(i).zfill(5)
            if train_mask[i]:
                folder = 'train'
            elif valid_mask[i]:
                folder = 'valid'
            else:
                folder = 'test'
            filename = os.path.splitext(os.path.basename(file_path))[0] + f'_{zid}.pt'
            path = os.path.join(processed_folder_path, folder, filename)
            torch.save(sample, path)
        logging.info(f"Skipped {skip_count} samples out of {i + 1}")
        logging.info(f"Trial length min/max: {trial_length_min_max}")
    
    # save session_list as txt
    info_path = os.path.join(processed_folder_path, 'info.txt')
    with open(info_path, 'w') as f:
        for session_id, num_units in session_list:
            f.write(f'{session_id} {num_units}\n')
