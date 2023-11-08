"""Load data, processes it, delete un-needed attributes, save into sample chuncks."""
import collections
import os
import argparse
import logging
import torch
import torch
from sklearn.model_selection import train_test_split
import scipy
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm
import re
from pathlib import Path

import argparse
import collections
import datetime
import logging
import os
from pathlib import Path
from typing import List

import msgpack
import numpy as np
import torch
import yaml
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from kirby.data import (
    Channel,
    Data,
    IrregularTimeSeries,
    Probe,
    RegularTimeSeries,
    Interval,
)
from kirby.tasks.reaching import REACHING
from kirby.taxonomy import (
    ChunkDescription,
    DandisetDescription,
    Macaque,
    Output,
    RecordingTech,
    SessionDescription,
    SortsetDescription,
    Species,
    Stimulus,
    StringIntEnum,
    SubjectDescription,
    Task,
    TrialDescription,
    to_serializable,
)
from kirby.utils import find_files_by_extension, make_directory


logging.basicConfig(level=logging.INFO)


WINDOW_SIZE = 1.0
STEP_SIZE = 0.5
JITTER_PADDING = 0.25


def extract_info_from_filename(filename):
    # Regular expression patterns for each component
    animal_name_pattern = re.compile(r'(Mihili|Chewie|MrT|Jaco)')
    task_type_pattern = re.compile(r'(CO|RT)')
    recording_date_pattern = re.compile(r'(\d{8})')

    # Search for each component in the filename
    animal_name_match = animal_name_pattern.search(filename)
    task_type_match = task_type_pattern.search(filename)
    recording_date_match = recording_date_pattern.search(filename)

    # If all components are found, extract and reformat them
    if animal_name_match and task_type_match and recording_date_match:
        animal_name = animal_name_match.group(0)
        task_type = task_type_match.group(0)
        recording_date = recording_date_match.group(0)

        # Decide the format based on the position relative to task_type in the filename
        if filename.index(recording_date) < filename.index(task_type):
            # Assume yyyymmdd format
            recording_date = f"{recording_date[0:4]}{recording_date[4:6]}{recording_date[6:8]}"
        else:
            # Assume mmddyyyy format
            recording_date = f"{recording_date[4:]}{recording_date[0:2]}{recording_date[2:4]}"

        return animal_name.lower(), task_type, recording_date
    else:
        raise ValueError(f"Unexpected filename: {filename}")


def extract_session_metadata(mat_dict):
    keys = mat_dict['data'][0][0][0][0].dtype.names
    values = mat_dict['data'][0][0][0][0][0]
    meta_dict = dict(zip(keys, values))

    start, end = meta_dict['dataWindow'][0]
    return float(start), float(end)


def extract_spikes(mat_dict: dict, prefix: str):
    r"""This dataset has spike sorted units."""
    units = mat_dict['data'][0][0][3][0]

    spikes = []
    names = []
    types = []
    waveforms = []
    unit_meta = []
    areas = set()

    # Map from common names to brodmann areas
    bas = {"pmd": 6, "m1": 4}
    
    for i in range(len(units)):
        # get unit meta data
        channel_id, sorted_id = units[i][0][0]
        array = units[i][1][0]
        bank = units[i][2][0]
        pin = units[i][3][0][0]
        channel_label = units[i][4][0]
        if len(units[i][5]) == 0:
            electrode_row = electrode_col = np.nan
        else:
            electrode_row = units[i][5][0][0]
            electrode_col = units[i][6][0][0]
        
        # note: {prefix}/{channel_label}/sorted_{sorted_id} does not appear to be unique
        # we temporarily add the index i to garantee uniqueness in the unit_name
        unit_name = f"{prefix}/{channel_label}/sorted_{sorted_id}/unit_{i}"

        # get spiketimes
        spiketimes = units[i][7][0][0][2][0, 0][:, 0]
        spikes.append(spiketimes)
        names.append([unit_name] * len(spiketimes))
        types.append(np.ones_like(spiketimes) * int(RecordingTech.UTAH_ARRAY_SPIKES))

        # get waveforms
        wf = units[i][7][0][0][2][0, 1][:]  # 48d waveform
        waveforms.append(wf)

        unit_meta.append(
            {
                "count": len(spiketimes),
                "channel_name": channel_label,
                "array": array,
                "pin": pin,
                "electrode_row": electrode_row,
                "electrode_col": electrode_col,
                "unit_name": unit_name,
                "area_name": array,
                "channel_number": channel_id,
                "unit_number": i,
                "ba": bas[array.lower()],
                "type": int(RecordingTech.UTAH_ARRAY_SPIKES),
                "average_waveform": wf.mean(axis=0),
            }
        )

        areas.add(array.lower())

    spikes = np.concatenate(spikes)
    waveforms = np.concatenate(waveforms)
    names = np.concatenate(names)
    types = np.concatenate(types)
    
    # Cast to torch tensors
    unit_meta_long = {}
    for key, item in unit_meta[0].items():
        stacked_array = np.stack([x[key] for x in unit_meta], axis=0)
        if np.issubdtype(type(item), np.number):
            if np.issubdtype(type(item), np.unsignedinteger):
                stacked_array = stacked_array.astype(np.int64)
            unit_meta_long[key] = torch.tensor(stacked_array)
        else:
            unit_meta_long[key] = stacked_array

    sorted = np.argsort(spikes)
    spikes = spikes[sorted]
    waveforms = waveforms[sorted]
    names = names[sorted]
    types = types[sorted]
    
    spikes = IrregularTimeSeries(
        timestamps=torch.tensor(spikes),
        waveforms=torch.tensor(waveforms),
        names=names,
        types=torch.tensor(types),
    )

    units = Data(**unit_meta_long)
    return spikes, units, list(areas)


def extract_behavior(mat_dict):
    r"""This dataset has cursor position, velocity, and acceleration.
    The cursor is controlled by the monkey using a manipulandum."""
    t, x, y, vx, vy, ax, ay = mat_dict['data'][0][0][2][0, 0]
    behavior = IrregularTimeSeries(
        timestamps=torch.tensor(t[:, 0]),
        cursor_pos=torch.tensor(np.concatenate([x, y], axis=1)),
        cursor_vel=torch.tensor(np.concatenate([vx, vy], axis=1)) 
        / 20., # todo: improve this so it does not have to be hard coded.
        cursor_acc=torch.tensor(np.concatenate([ax, ay], axis=1)), )
    return behavior


def extract_trial_metadata_co(mat_dict: dict, behavior: IrregularTimeSeries):
    r"""In the center-out task, after a short hold period the monkey has to move 
    the cursor from the center to one of the targets, and then return to the center.
    The trial ends when the hold period of next trial starts. There are three phases: 
    hold, reach, return."""
    # get trial table
    keys = mat_dict['data'][0][0][1][0].dtype.names
    values = mat_dict['data'][0][0][1][0][0]
    
    # the trial starts when the target is shown on the screen
    start_of_trial = values['tgtOnTime'][:, 0]
    # the end is when the target for the next trial is shown
    start_of_next_trial = np.append(start_of_trial[1:], np.nan)
    # When there is no next trial (nan value), we use the end of the trial plus 1s
    default_value = values['endTime'][:, 0] + 1.
    nan_mask = np.isnan(start_of_next_trial)
    start_of_next_trial[nan_mask] = default_value[nan_mask]

    trials = Interval(
        start=torch.tensor(start_of_trial),
        end=torch.tensor(start_of_next_trial),
        target_on_time=torch.tensor(values['tgtOnTime'][:, 0]),
        go_cue_time=torch.tensor(values['goCueTime'][:, 0]),
        target_acq_time=torch.tensor(values['endTime'][:, 0]),
        target_id=torch.tensor(values['tgtID'][:, 0]),
        target_dir=torch.tensor(values['tgtDir'][:, 0]),
        result=values['result'],
    )

    # behavior_type is a segmentation map that indicates which period of the trial we are in.
    behavior_type = torch.ones_like(behavior.timestamps, dtype=torch.long) * REACHING.RANDOM
    # stimuli events
    stimuli_timestamps = []
    stimuli_type = []
    # stimuli segments
    stimuli_segment_start = []
    stimuli_segment_end = []
    stimuli_reach_direction_id = []
    stimuli_reach_direction_deg = []

    for i in range(len(trials)):
        # first we check whether the trial was successful, and is valid.        
        if check_co_baseline_trial_validity(trials[i]):
            behavior_type[(behavior.timestamps >= trials.target_on_time[i]) 
                          & (behavior.timestamps < trials.go_cue_time[i])] = REACHING.CENTER_OUT_HOLD
            behavior_type[(behavior.timestamps >= trials.go_cue_time[i]) 
                          & (behavior.timestamps < trials.target_acq_time[i])] = REACHING.CENTER_OUT_REACH
            behavior_type[(behavior.timestamps >= trials.target_acq_time[i]) 
                          & (behavior.timestamps < trials.end[i])] = REACHING.CENTER_OUT_RETURN
            
            # each trial will have two reach directions, one during the reach phase, 
            # and one during the return phase (the opposite direction)
            stimuli_segment_start.extend([trials.go_cue_time[i], trials.target_acq_time[i]])
            stimuli_segment_end.extend([trials.target_acq_time[i], trials.end[i]])
            stimuli_reach_direction_id.extend([trials.target_id[i], np.mod(trials.target_id[i], 4)])
            stimuli_reach_direction_deg.extend([trials.target_dir[i], np.mod(trials.target_dir[i], 360.)])
        else:
            # mark all corresponding timestamps as invalid
            behavior_type[(behavior.timestamps >= trials.target_on_time[i]) & (behavior.timestamps < trials.end[i])] = REACHING.INVALID

        # collect stimuli events: target shown, go cue, target acquired, note that some of these values will be nan
        stimuli_timestamps.extend([trials.target_on_time[i], trials.go_cue_time[i], trials.target_acq_time[i]])
        stimuli_type.extend([REACHING.TARGET_ON, REACHING.GO_CUE, REACHING.TARGET_ACQUIRED])
    
    behavior.type = behavior_type

    sort_idx = np.argsort(stimuli_timestamps)
    stimuli_timestamps = np.array(stimuli_timestamps)[sort_idx]
    stimuli_type = np.array(stimuli_type)[sort_idx]

    valid_idx = ~np.isnan(stimuli_timestamps)

    stimuli_events = IrregularTimeSeries(
        timestamps=torch.tensor(stimuli_timestamps[valid_idx]),
        type=torch.tensor(stimuli_type[valid_idx])
        )
    
    stimuli_segments = Interval(
        start=torch.tensor(stimuli_segment_start),
        end=torch.tensor(stimuli_segment_end),
        reach_direction_id=torch.tensor(stimuli_reach_direction_id),
        reach_direction_deg=torch.tensor(stimuli_reach_direction_deg)
        )
    
    return trials, behavior, stimuli_events, stimuli_segments


def extract_trial_metadata_rt(mat_dict: dict, behavior: IrregularTimeSeries):
    r"""In the random target task, after a short hold period the monkey has to move 
    the cursor across multiple random targets."""
    # get trial table
    keys = mat_dict['data'][0][0][1][0].dtype.names
    values = mat_dict['data'][0][0][1][0][0]
    
    trials = Interval(
        start=torch.tensor(values['startTime'][:, 0]),
        end=torch.tensor(values['endTime'][:, 0]),
        go_cue_time_1=torch.tensor(values['goCueTime'][:, 0]),
        go_cue_time_2=torch.tensor(values['goCueTime'][:, 1]),
        go_cue_time_3=torch.tensor(values['goCueTime'][:, 2]),
        go_cue_time_4=torch.tensor(values['goCueTime'][:, 3]),
        result=values['result'],
        num_attempts=torch.tensor(values['numAttempted'][:, 0]),
        )
    
    # behavior_type is a segmentation map that indicates which period of the trial we are in.
    behavior_type = torch.ones_like(behavior.timestamps, dtype=torch.long) * REACHING.RANDOM
    # stimuli events
    stimuli_timestamps = []
    stimuli_type = []
    # stimuli segments
    # todo: estimate the reach direction in degrees based on the angle between consecutive targets
    stimuli_segment_start = []
    stimuli_segment_end = []
    stimuli_reach_direction_deg = []

    for i in range(len(trials)):
        # there is a short hold period before the first go_cue is given.
        behavior_type[(behavior.timestamps >= trials.start[i]) 
                      & (behavior.timestamps < trials.go_cue_time_1[i])] = REACHING.HOLD
        behavior_type[(behavior.timestamps >= trials.go_cue_time_1[i]) 
                      & (behavior.timestamps < trials.end[i])] = REACHING.RANDOM
        
        stimuli_timestamps.extend([trials.go_cue_time_1[i], trials.go_cue_time_2[i], trials.go_cue_time_3[i], trials.go_cue_time_4[i]])
        stimuli_type.extend([REACHING.GO_CUE, REACHING.GO_CUE, REACHING.GO_CUE, REACHING.GO_CUE])

    behavior.type = behavior_type
    
    sort_idx = np.argsort(stimuli_timestamps)
    stimuli_timestamps = np.array(stimuli_timestamps)[sort_idx]
    stimuli_type = np.array(stimuli_type)[sort_idx]

    valid_idx = ~np.isnan(stimuli_timestamps)

    stimuli_events = IrregularTimeSeries(
        timestamps=torch.tensor(stimuli_timestamps[valid_idx]),
        type=torch.tensor(stimuli_type[valid_idx])
        )
        
    stimuli_segments = None
    
    return trials, behavior, stimuli_events, stimuli_segments


def detect_outliers(data: Data, filepath: str):
    r"""Detect outliers in the behavior, based on the hand acceleration and the
    position of the hand. The monkeys get frustrated and start banging the manipulandum
    which leads to high acceleration values, and for the cursor to go outside the screen.
    
    TODO: For the public release of the data, we will add this information to the data.
    """

    def identify_outliers(data, threshold=1500):
        # identify outliers based on hand acceleration
        hand_acc_norm = np.linalg.norm(data.behavior.cursor_acc, axis=1)
        mask = hand_acc_norm > threshold
        structure = np.array([1, 1], dtype=bool)
        # Dilate the binary mask
        dilated = binary_dilation(mask, structure)
        return dilated

    def identify_outliers_box(data, x0, x1, y0, y1):
        # identify outliers outside the screen bounds
        hand_pos = data.behavior.cursor_pos.numpy()
        mask = np.logical_or(hand_pos[:, 0] < x0, hand_pos[:, 0] > x1) 
        mask = np.logical_or(mask, hand_pos[:, 1] < y0)
        mask = np.logical_or(mask, hand_pos[:, 1] > y1)

        structure = np.ones(400, dtype=bool)
        # Dilate the binary mask
        dilated = binary_dilation(mask, structure)

        structure = np.ones(100, dtype=bool)
        # Erode the binary mask
        eroded = binary_erosion(dilated, structure)

        return eroded

    # some files have a different screen size, so we need to adjust the bounds
    special_1 = [
        "./raw/Wave2/Chewie_CO_FF_BL_09152016_001_stripped.mat",
        "./raw/Wave2/Chewie_CO_VR_BL_09122016_001_stripped.mat",
        "./raw/Wave2/Chewie_CO_VR_BL_09142016_001_stripped.mat",
        "./raw/Wave2/Chewie_CO_FF_BL_09192016_001_stripped.mat",
        "./raw/Wave2/Chewie_CO_FF_BL_09212016_001_stripped.mat",
        "./raw/Wave2/Chewie_CO_VR_BL_09092016_001_stripped.mat",
        ]

    special_2 = [
    "./raw/Wave2/Jaco_CO_FF_BL_04062016_001_stripped.mat",
    "./raw/Wave2/Jaco_CO_FF_BL_04052016_001_stripped.mat",
]

    if filepath in special_1:
        x0, x1, y0, y1 = -5, 10, -30, -15  
    elif filepath in special_2:
        x0, x1, y0, y1 = -5, 18, -46, -24 
    else:
        x0, x1, y0, y1 = -10, 15, -45, -20
    # remove high accerleration
    mask_acc = identify_outliers(data, threshold=1500)
    # remove outliers outside the screen bounds
    mask_rect = identify_outliers_box(data, x0, x1, y0, y1)
    mask = np.logical_or(mask_acc, mask_rect)
    return mask


def filter_buckets(buckets):
    out = []
    for bucket in buckets:
        # count percentage of outliers
        outlier_ratio = torch.sum(bucket.behavior.type == REACHING.OUTLIER).item() / len(bucket.behavior)
        if outlier_ratio < 0.25:
            out.append(bucket)
    return out


def check_co_baseline_trial_validity(trial, min_duration=0.5, max_duration=6.0):
    # check if the trial was successful
    cond1 = trial['result'] == 'R'
    cond2 = not trial['target_id'].isnan()

    # check if the duration of the trial is between min_duration and max_duration
    cond3 = (trial['end'] - trial['target_on_time']) > min_duration and (trial['end'] - trial['target_on_time']) < max_duration
    return all([cond1, cond2, cond3])


def check_rt_baseline_trial_validity(trial, min_duration=2.0, max_duration=10.0):
    # check if the trial was successful
    cond1 = trial['result'] == 'R'
    cond2 = trial['num_attempts'] == 4

    # check if the duration of the trial is between min_duration and max_duration
    cond3 = (trial['end'] - trial['start']) > min_duration and (trial['end'] - trial['start']) < max_duration
    return all([cond1, cond2, cond3])


def split_and_get_train_valid_test(trials, test_size=0.2, valid_size=0.1, random_state=42):
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


def encode_datetime(obj):
    """msgpack doesn't support datetime, so we need to encode it as a string."""
    if isinstance(obj, datetime.datetime):
        return obj.strftime("%Y%m%dT%H:%M:%S.%f").encode()


if __name__ == "__main__":
    experiment_name = "perich_miller_population_2018"

    # Use argparse to extract two arguments from the command line:
    # input_dir and output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()
    raw_folder_path = args.input_dir
    processed_folder_path = args.output_dir

    make_directory(processed_folder_path, prompt_if_exists=False)
    make_directory(os.path.join(processed_folder_path, 'train'))
    make_directory(os.path.join(processed_folder_path, 'valid'))
    make_directory(os.path.join(processed_folder_path, 'test'))

    # todo this is not good naming, fix it. needed for few-shot finetuning 
    # where we know the segments where there is data (during actually trials)
    make_directory(os.path.join(processed_folder_path, 'finetune'))

    # the files that have a _stripped suffix have been processed (by M.P.) 
    # to make them readable in Python
    extension = "_stripped.mat"
    session_list = []

    # Here, we will have multiple trials in each session
    sortsets = {}
    trials: list[TrialDescription] = []

    # We don't have any info about age or sex for these subjects.
    # todo look into the papers, check if these are the right species
    subjects = [
        SubjectDescription(id="mihili", species=Species.MACACA_MULATTA),
        SubjectDescription(id="chewie", species=Species.MACACA_MULATTA),
        SubjectDescription(id="jaco", species=Species.MACACA_MULATTA),
        SubjectDescription(id="mrt", species=Species.MACACA_MULATTA),
    ]

    known_sessions = {}
    
    # find all files with extension .nwb in folder_path
    for file_path in tqdm(sorted(find_files_by_extension(raw_folder_path, extension))):
        if not "BL" in file_path:
            # for now, we will skip files that correspond to perturbation sessions
            # and only use the baseline experiments.
            continue

        logging.info(f"Processing file: {file_path}")
        
        # determine session_id and sortset_id
        animal, task, recording_date = extract_info_from_filename(Path(file_path).stem)
        session_id = f"{animal}_{recording_date}_{task}"

        # There's one instance where there's a duplicate session id (chewie_20160929_CO)
        # we skip it.
        if session_id in known_sessions:
            continue

        known_sessions[session_id] = True

        # spike sorting was done on all data from the same day, so we should 
        # have the same neurons/units.
        sortset_id = f"{animal}_{recording_date}"
        
        # load file
        mat_dict = scipy.io.loadmat(file_path)

        # extract session start and end times
        session_start, session_end = extract_session_metadata(mat_dict)
        
        # extract spiking activity
        spikes, units, areas = extract_spikes(mat_dict, sortset_id)

        # extract behavior
        behavior = extract_behavior(mat_dict)

        # extract trial structure relative to each task
        if task == "CO":
            trials, behavior, stimuli_events, stimuli_segments = extract_trial_metadata_co(mat_dict, behavior)
        elif task == "RT":
            trials, behavior, stimuli_events, stimuli_segments = extract_trial_metadata_rt(mat_dict, behavior)
        else:
            raise ValueError("Unknown task type")

        # sanity checkpoint: check that recording started at 0
        first_recorded_point = min(behavior.timestamps[0], spikes.timestamps[0])
        last_recorded_point = max(behavior.timestamps[-1], spikes.timestamps[-1])
        assert abs(first_recorded_point - session_start) < 10 and abs(last_recorded_point - session_end) < 10

        data = Data(
            start=session_start,
            end=session_end,
            spikes=spikes,
            units=units,
            behavior=behavior,
            trials=trials,
            stimuli_events=stimuli_events,
            stimuli_segments=stimuli_segments,
            session=session_id,
            sortset=sortset_id,
            subject=animal,
            # todo: add probes
        )

        # there are some outliers in the behavior, that we will need to remove
        mask = detect_outliers(data, file_path)
        data.behavior.type[mask] = REACHING.OUTLIER

        # split data into train, validation, and test
        # but first we will to identify trials that are valid/successful
        if "CO" in file_path:
            successful_trials = list(filter(check_co_baseline_trial_validity, data.trials))
        elif "RT" in file_path:
            successful_trials = list(filter(check_rt_baseline_trial_validity, data.trials))

        logging.info(f"Found {len(successful_trials)} out of {len(data.trials)} trials that were successful.")
        (
            train_trials,
            valid_trials,
            test_trials,
        ) = split_and_get_train_valid_test(successful_trials)

        # collect data slices for validation and test trials
        train_slices = collect_slices(data, train_trials)
        valid_slices = collect_slices(data, valid_trials)
        test_slices = collect_slices(data, test_trials)

        # the remaining data (unstructured) is used for training
        train_buckets = list(data.bucketize(WINDOW_SIZE, STEP_SIZE, JITTER_PADDING))
        # we make sure to exclude validation and test data from the training buckets
        train_buckets = exclude_from_train(train_buckets, valid_trials + test_trials)
        # remove buckets where there are a lot of outliers
        train_buckets = filter_buckets(train_buckets)

        chunks = collections.defaultdict(list)
        footprints = collections.defaultdict(list)

        logging.info("Saving to disk.")
        for buckets, fold in [
            (train_buckets, "train"),
            (train_slices, "finetune"),
            (valid_slices, "valid"),
            (test_slices, "test"),
        ]:
            for i, sample in enumerate(buckets):
                basename = f"{session_id}_{i:05}"
                filename = f"{basename}.pt"
                path = os.path.join(processed_folder_path, fold, filename)

                # precompute map from unit name to indices of that unit in data.spikes
                sample.spikes.precompute_index_map(field='names')
                torch.save(sample, path)

                footprints[fold].append(os.path.getsize(path))
                chunks[fold].append(
                    ChunkDescription(
                        id=basename,
                        duration=(sample.end - sample.start).item(),
                        start_time=sample.start.item(),
                    )
                )
        
        footprints = {k: int(np.mean(v)) for k, v in footprints.items()}

        # Create the metadata for description.yaml
        # Get the channel names from the regular file.
        area_map = {
            "m1": Macaque.primary_motor_cortex,
            "pmd": Macaque.premotor_cortex,
        }
        if sortset_id not in sortsets:
            # Verify which areas are present in this sortset.
            areas = [area_map[x] for x in areas]

            sortsets[sortset_id] = SortsetDescription(
                id=sortset_id,
                subject=animal,
                areas=areas,
                recording_tech=[RecordingTech.UTAH_ARRAY_SPIKES, RecordingTech.UTAH_ARRAY_WAVEFORMS],
                sessions=[],
                units=[],
                )

        session = SessionDescription(
            id=session_id,
            start_time=datetime.datetime.strptime(recording_date, "%Y%m%d"),
            end_time=datetime.datetime.strptime(recording_date, "%Y%m%d")
            + datetime.timedelta(seconds=session_end - session_start),
            task=Task.DISCRETE_REACHING,
            inputs={
                RecordingTech.UTAH_ARRAY_SPIKES: "spikes",
                RecordingTech.UTAH_ARRAY_WAVEFORMS: "spikes.waveforms",
            },
            stimuli={
                Stimulus.GO_CUE: "stimuli_events.go_cue"
                },
            outputs={
                Output.CURSOR2D: "behavior.cursor_vel",
            },
            trials=[],
            )

        # todo unclear what the definition of trial is.
        trial = TrialDescription(
            id=f"{experiment_name}_{session_id}_01",
            chunks=chunks,
            footprints=footprints,
        )
        session.trials = [trial]

        sortsets[sortset_id].units.append(units.unit_name)
        sortsets[sortset_id].sessions.append(session)


    # Transform sortsets to a list of lists, otherwise it won't serialize to yaml.
    sortsets = sorted(list(sortsets.values()), key=lambda x: x.id)
    for x in sortsets:
        x.units = sorted(list(set(np.concatenate(x.units).tolist())))

    # Create a description file for ease of reference.
    description = DandisetDescription(
        id=experiment_name,
        origin_version="0.0.1",  # Not public yet
        derived_version="0.0.1",  # This variant
        metadata_version="0.0.1",
        source="private",
        description="Reaching dataset from Perich et al. (2018), data from M1 and PMd.",
        folds=["train", "valid", "test"],
        subjects=subjects,
        sortsets=sortsets,
    )

    # Efficiently encode enums to strings
    description = to_serializable(description)

    filename = Path(processed_folder_path) / "description.yaml"
    print(f"Saving description to {filename}")

    with open(filename, "w") as f:
        yaml.dump(description, f)

    # For efficiency, we also save a msgpack version of the description.
    # Smaller on disk, faster to read.
    filename = Path(processed_folder_path) / "description.mpk"
    print(f"Saving description to {filename}")

    with open(filename, "wb") as f:
        msgpack.dump(description, f, default=encode_datetime)
