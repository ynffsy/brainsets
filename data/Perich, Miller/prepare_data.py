"""Load data, processes it, delete un-needed attributes, save into sample chuncks."""
import os
import logging
import torch
import torch
from sklearn.model_selection import train_test_split
import scipy
import numpy as np

from kirby.utils import find_files_by_extension, make_directory
from kirby.data import Data, IrregularTimeSeries, Interval
from kirby.tasks.reaching import REACHING


logging.basicConfig(level=logging.INFO)


WINDOW_SIZE = 1.0
STEP_SIZE = 0.5
JITTER_PADDING = 0.25


# Trial outcomes
# R = REWARD
# A = ABORT, target is not shown, go cue is never shown
# F = FAIL
# I = INCOMPLETE

def load_file(file_path):
    # load mat file
    mat_dict = scipy.io.loadmat(file_path)

    # session metadata
    start, end = extract_session_metadata(mat_dict)
    
    # process behavior, should include hand position, velocity, acceleration
    behavior = extract_behavior(mat_dict)

    # trial table
    if "CO" in file_path:
        trials, behavior, target_direction, go_cue, target_acquired = extract_co_baseline_trials(mat_dict, behavior)
    elif "RT" in file_path:
        trials, behavior, go_cue = extract_rt_baseline_trials(mat_dict, behavior)
    else:
        raise ValueError("Unknown task type")

    # unit activity
    spikes, units = extract_spikes(mat_dict=mat_dict)

    # check that recording started at 0
    first_recorded_point = min(behavior.timestamps[0], spikes.timestamps[0])
    last_recorded_point = max(behavior.timestamps[-1], spikes.timestamps[-1])
    assert abs(first_recorded_point - start) < 10 and abs(last_recorded_point - end) < 10

    if "CO" in file_path:
        data = Data(spikes=spikes, behavior=behavior, trials=trials, units=units, start=start, end=end,
                    go_cue=go_cue, target_acquired=target_acquired, target_direction=target_direction,)
    elif "RT" in file_path:
        data = Data(spikes=spikes, behavior=behavior, trials=trials, units=units,
                    start=start, end=end, go_cue=go_cue)
    return data


def extract_session_metadata(mat_dict):
    keys = mat_dict['data'][0][0][0][0].dtype.names
    values = mat_dict['data'][0][0][0][0][0]
    meta_dict = dict(zip(keys, values))

    start, end = meta_dict['dataWindow'][0]
    return start, end


def extract_spikes(mat_dict):
    units = mat_dict['data'][0][0][3][0]

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

    spikes = IrregularTimeSeries(torch.tensor(spikes), unit_id=torch.tensor(unit_ids))
    units = Data(**unit_meta)
    return spikes, units


def extract_behavior(mat_dict):
    t, x, y, vx, vy, ax, ay = mat_dict['data'][0][0][2][0, 0]
    behavior = IrregularTimeSeries(
        timestamps=torch.tensor(t[:, 0]),
        hand_pos=torch.tensor(np.concatenate([x, y], axis=1)),
        hand_vel=torch.tensor(np.concatenate([vx, vy], axis=1)) / 20.,
        hand_acc=torch.tensor(np.concatenate([ax, ay], axis=1)), )
    return behavior


def extract_co_baseline_trials(mat_dict, behavior):
    # trial table
    keys = mat_dict['data'][0][0][1][0].dtype.names
    values = mat_dict['data'][0][0][1][0][0]
    
    # the trial = hold period + center-out reach period + return to center period
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
    behavior_type = torch.ones_like(timestamps, dtype=torch.long) * REACHING.RANDOM
    go_cue_event = []
    target_acquired_event = []
    target_direction_timestamps = []
    target_direction = []

    for i in range(len(trials)):
        success = check_co_baseline_trial_validity(trials[i])
        
        if success:
            behavior_type[(timestamps >= trials.target_on_time[i]) & (timestamps < trials.go_cue_time[i])] = REACHING.CENTER_OUT_HOLD
            behavior_type[(timestamps >= trials.go_cue_time[i]) & (timestamps < trials.end_time[i])] = REACHING.CENTER_OUT_REACH
            behavior_type[(timestamps >= trials.end_time[i]) & (timestamps < trials.end[i])] = REACHING.CENTER_OUT_RETURN
            if not np.isnan(trials.target_dir[i]):
                target_direction_timestamps.append((trials.go_cue_time[i] + trials.end_time[i]) * 0.5)
                target_direction.append(trials.target_dir[i])
        else:
            behavior_type[(timestamps >= trials.target_on_time[i]) & (timestamps < trials.end[i])] = REACHING.INVALID

        if not np.isnan(trials.go_cue_time[i]):
            go_cue_event.append(trials.go_cue_time[i])
        if not np.isnan(trials.end_time[i]):
            target_acquired_event.append(trials.end_time[i])
        
    behavior.type = behavior_type
    target_direction = IrregularTimeSeries(timestamps=torch.tensor(target_direction_timestamps), 
                                           direction=torch.tensor(target_direction))
    go_cue = IrregularTimeSeries(timestamps=torch.tensor(go_cue_event))
    target_acquired = IrregularTimeSeries(timestamps=torch.tensor(target_acquired_event))

    return trials, behavior, target_direction, go_cue, target_acquired


def extract_rt_baseline_trials(mat_dict, behavior):
    # trial table
    keys = mat_dict['data'][0][0][1][0].dtype.names
    values = mat_dict['data'][0][0][1][0][0]
    
    trials = Interval(start=torch.tensor(values['startTime'][:, 0]),
                        end=torch.tensor(values['endTime'][:, 0]),
                        # other events
                        start_time=torch.tensor(values['startTime'][:, 0]),
                        go_cue_time=torch.tensor(values['goCueTime'][:, 0]),
                        end_time=torch.tensor(values['endTime'][:, 0]),
                        result=values['result'],
                        num_attempts=torch.tensor(values['numAttempted'][:, 0]),
                        )
    
    go_cue_event = []
    for i in range(len(values['goCueTime'])):
        for j in range(4):
            if not np.isnan(values['goCueTime'][i, j]):
                go_cue_event.append(values['goCueTime'][i, j])

    timestamps = behavior.timestamps
    behavior_type = torch.ones_like(timestamps, dtype=torch.long) * REACHING.RANDOM
    for i in range(len(trials)):
        behavior_type[(timestamps >= trials.start_time[i]) & (timestamps < trials.go_cue_time[i])] = REACHING.HOLD
        behavior_type[(timestamps >= trials.go_cue_time[i]) & (timestamps < trials.end_time[i])] = REACHING.RANDOM

    behavior.type = behavior_type

    go_cue = IrregularTimeSeries(timestamps=torch.tensor(go_cue_event))
    return trials, behavior, go_cue


##############################
# Validation and test splits # 
##############################
def check_co_baseline_trial_validity(trial, min_duration=0.5, max_duration=6.0):
    # check if the trial was successful
    cond1 = trial['result'] == 'R'
    cond2 = not trial['target_id'].isnan()

    # check if the duration of the trial is between min_duration and max_duration
    cond3 = (trial['end'] - trial['start']) > min_duration and (trial['end'] - trial['start']) < max_duration
    return all([cond1, cond2, cond3])


def check_rt_baseline_trial_validity(trial, min_duration=2.0, max_duration=10.0):
    # check if the trial was successful
    cond1 = trial['result'] == 'R'
    cond2 = trial['num_attempts'] == 4

    # check if the duration of the trial is between min_duration and max_duration
    cond3 = (trial['end'] - trial['start']) > min_duration and (trial['end'] - trial['start']) < max_duration
    return all([cond1, cond2, cond3])



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


#######################
# Estimate input size #
#######################
def next_power_of_2(x):
    return 1<<(int(x)-1).bit_length()

def get_num_input_tokens(data, bucket_size, jitter):
    def _get_num_input_tokens(data):
        return len(data.spikes)
    bin_size = 0.5
    bins = np.arange(0, bucket_size + 2 * jitter, bin_size)
    input_size_in_bins = []
    for start in bins:
        end = start + bin_size
        input_size_in_bins.append(_get_num_input_tokens(data.slice(start, end)))
    input_size_in_bins = np.array(input_size_in_bins)
    input_sizes = np.convolve(input_size_in_bins,np.ones(int(np.ceil(bucket_size / bin_size)), dtype=int), 'valid')
    total_input_size = input_sizes.max() + len(data.units.id) * 2
    total_input_size += len(data.units.id)  # since this is an approximation, we add the number of units
    return total_input_size


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
            # only use baseline files
            continue
        logging.info(f"Processing file: {file_path}")
        data = load_file(file_path)

        # get successful trials, and keep 20% for test, 10% for validation
        if "CO" in file_path:
            valid_trials = list(filter(check_co_baseline_trial_validity, data.trials))
        elif "RT" in file_path:
            valid_trials = list(filter(check_rt_baseline_trial_validity, data.trials))
        validation_trials, test_trials = split_and_get_validation_test(valid_trials, test_size=0.2, valid_size=0.1, 
                                                                       random_state=42)

        # collect data slices for validation and test trials
        validation_slices = collect_slices(data, validation_trials)
        test_slices = collect_slices(data, test_trials)

        # the remaining data (unstructured) is used for training
        train_buckets = list(data.bucketize(WINDOW_SIZE, STEP_SIZE, JITTER_PADDING))
        # we make sure to exclude validation and test data from the training buckets
        train_buckets = exclude_from_train(train_buckets, validation_trials + test_trials)

        # for each bucket we estimate the expected number of input tokens
        for bucket in train_buckets:
            max_num_input_tokens = get_num_input_tokens(train_buckets[1], WINDOW_SIZE, JITTER_PADDING)  # 3233, 3249, 3249, 3265, 3271, 3272, 3273, 3283, 3296, 3353
            batch_bucket_size = next_power_of_2(max_num_input_tokens)
            bucket.max_num_input_tokens = max_num_input_tokens
            bucket.batch_bucket_size = batch_bucket_size

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

    # save session_list as txt
    info_path = os.path.join(processed_folder_path, 'info.txt')
    with open(info_path, 'w') as f:
        for session_id, num_units in session_list:
            f.write(f'{session_id} {num_units}\n')
