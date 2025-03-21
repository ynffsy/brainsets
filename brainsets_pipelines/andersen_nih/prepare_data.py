import argparse
import datetime
import logging
import h5py
import os

import numpy as np
from pynwb import NWBHDF5IO
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion

from temporaldata import Data, IrregularTimeSeries, Interval
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    DeviceDescription,
)
from brainsets.utils.dandi_utils import (
    extract_spikes_from_nwbfile,
    extract_subject_from_nwb,
)
from brainsets.taxonomy import RecordingTech, ImplantArea, Task
from brainsets import serialize_fn_map
import ipdb
import matplotlib.pyplot as plt



logging.basicConfig(level=logging.INFO)



def extract_behavior(nwbfile):
    """Extract behavior from the NWB file.

    ..note::
        Cursor position and target position are in the same frame of reference.
        They are both of size (sequence_len, 2).
    """

    timestamps = nwbfile.acquisition['cursor_pos'].timestamps[:]
    cursor_pos = nwbfile.acquisition['cursor_pos'].data[:]  # 2d
    cursor_vel = nwbfile.acquisition['cursor_vel'].data[:]
    target_pos = nwbfile.acquisition['target_pos'].data[:]
    cursor_acc = np.gradient(cursor_vel, axis=0)

    # remove rows where target_pos is nan
    mask = np.isnan(target_pos).any(axis=1)
    rows_with_nan = np.where(mask)[0]
    
    timestamps = np.delete(timestamps, rows_with_nan)
    cursor_pos = np.delete(cursor_pos, rows_with_nan, axis=0)
    cursor_vel = np.delete(cursor_vel, rows_with_nan, axis=0)
    target_pos = np.delete(target_pos, rows_with_nan, axis=0)
    cursor_acc = np.delete(cursor_acc, rows_with_nan, axis=0)

    direction_to_target = target_pos - cursor_pos

    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_pos,
        vel=cursor_vel,
        acc=cursor_acc,
        direction_to_target=direction_to_target,
        target_pos=target_pos,
        domain="auto",
    )

    print(
        f'cursor pos (mean/std): {np.abs(np.mean(cursor_pos)):.4f}/{np.std(cursor_pos):.4f} '
        f'cursor vel (mean/std): {np.abs(np.mean(cursor_vel)):.4f}/{np.std(cursor_vel):.4f} ' 
        f'cursor acc (mean/std): {np.abs(np.mean(cursor_acc)):.4f}/{np.std(cursor_acc):.4f} ' 
        f'target pos (mean/std): {np.abs(np.mean(target_pos)):.4f}/{np.std(target_pos):.4f} ' 
        f'direction to target (mean/std): {np.abs(np.mean(direction_to_target)):.4f}/{np.std(direction_to_target):.4f}')

    return cursor


def visualize_behavior(nwbfile, cursor):

    plt.figure(figsize=(10, 10))

    # plt.plot(cursor.pos[:, 0], cursor.pos[:, 1], label='cursor position')
    # plt.plot(cursor.target_pos[:, 0], cursor.target_pos[:, 1], label='target position')
    plt.plot(cursor.direction_to_target[:, 0], cursor.direction_to_target[:, 1], label='direction to target')


    plt.show()

    ipdb.set_trace()


def extract_trials(nwbfile, task, cursor):
    r"""Extract trial information from the NWB file. Trials that are flagged as
    "to discard" or where the monkey failed are marked as invalid."""
    trial_table = nwbfile.trials.to_dataframe()

    # rename start and end time columns
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
        }
    )
    trials = Interval.from_dataframe(trial_table)

    # next we extract the different periods in the trials
    if task == "CenterStart" or task == "RadialGrid":
        # isolate valid trials based on success
        trials.is_valid = np.logical_and(
            ~(np.isnan(trials.first_contact_time)),
            trials.first_contact_time < 4.0,
        )
        valid_trials = trials.select_by_mask(trials.is_valid)

        movement_phases = Data(
            reach_period=Interval(
                start=valid_trials.go_time, 
                end=valid_trials.go_time + valid_trials.first_contact_time),
            hold_period=Interval(
                start=valid_trials.go_time + valid_trials.first_contact_time, 
                end=valid_trials.end),
            domain="auto",
        )
    
    elif task == "CenterOut":
        # isolate valid trials based on success
        # also remove CAR trials

        # assist_mask = (trials.assist_level == 0) # Assist 0 only
        assist_mask = (np.abs(trials.assist_level) < 0.35) # Low assist
        # assist_mask = (trials.assist_level > -0.5) # Exclude CAR trials
        # assist_mask = (np.abs(trials.assist_level) < 1) # Exclude CAR and open loop trials

        trials.is_valid = np.logical_and.reduce((
            ~(np.isnan(trials.first_contact_time)),
            assist_mask,
            trials.first_contact_time < 4.0,
            trials.target_index != 9,
        ))
        valid_trials = trials.select_by_mask(trials.is_valid)

        movement_phases = Data(
            reach_period=Interval(
                start=valid_trials.go_time, 
                end=valid_trials.go_time + valid_trials.first_contact_time),
            domain="auto",
        )

        # movement_phases = Data(
        #     reach_period=Interval(
        #         start=valid_trials.go_time, 
        #         end=valid_trials.go_time + 0.5),
        #     domain="auto",
        # )

    # everything outside of the different identified periods will be marked as random
    movement_phases.random_period = cursor.domain.difference(movement_phases.domain)

    return trials, movement_phases


# def detect_outliers(cursor):
#     # sometimes monkeys get angry, we want to identify the segments where the hand is
#     # moving too fast, and mark them as outliers
#     # we use the norm of the acceleration to identify outliers
#     hand_acc_norm = np.linalg.norm(cursor.acc, axis=1)
#     mask_acceleration = hand_acc_norm > 1500.0
#     mask_acceleration = binary_dilation(
#         mask_acceleration, structure=np.ones(2, dtype=bool)
#     )

#     # we also want to identify out of bound segments
#     mask_position = np.logical_or(cursor.pos[:, 0] < -10, cursor.pos[:, 0] > 10)
#     mask_position = np.logical_or(mask_position, cursor.pos[:, 1] < -10)
#     mask_position = np.logical_or(mask_position, cursor.pos[:, 1] > 10)
#     # dilate than erode
#     mask_position = binary_dilation(mask_position, np.ones(400, dtype=bool))
#     mask_position = binary_erosion(mask_position, np.ones(100, dtype=bool))

#     outlier_mask = np.logical_or(mask_acceleration, mask_position)

#     # convert to interval, you need to find the start and end of the outlier segments
#     start = cursor.timestamps[np.where(np.diff(outlier_mask.astype(int)) == 1)[0]]
#     if outlier_mask[0]:
#         start = np.insert(start, 0, cursor.timestamps[0])

#     end = cursor.timestamps[np.where(np.diff(outlier_mask.astype(int)) == -1)[0]]
#     if outlier_mask[-1]:
#         end = np.insert(end, 0, cursor.timestamps[-1])

#     cursor_outlier_segments = Interval(start=start, end=end)

#     return cursor_outlier_segments


def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--brainset_id", type=str, default="andersen_nih")
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./processed")
    parser.add_argument("--array",      type=str, default=None)

    args = parser.parse_args()

    # intiantiate a DatasetBuilder which provides utilities for processing data
    brainset_description = BrainsetDescription(
        id=args.brainset_id,
        origin_version="",
        derived_version="",
        source="",
        description="",
    )

    logging.info(f"Processing file: {args.input_file}")

    # open file
    io = NWBHDF5IO(args.input_file, "r")
    nwbfile = io.read()

    # extract subject metadata
    # this dataset is from dandi, which has structured subject metadata, so we
    # can use the helper function extract_subject_from_nwb
    subject = extract_subject_from_nwb(nwbfile)

    # extract experiment metadata
    recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
    if args.array is not None:
        device_id = f"{subject.id}_{args.array}_{recording_date}"
    else:
        device_id = f"{subject.id}_{recording_date}"

    # extract the task from the file name
    file_path = io._file.filename
    file_path_no_ext = file_path.rsplit('.', 1)[0]
    task = file_path_no_ext.rsplit('_', 1)[-1]

    session_id = f"{device_id}_{task}"

    # register session
    session_description = SessionDescription(
        id=session_id,
        recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
        task=Task.REACHING,
    )

    # register device
    device_description = DeviceDescription(
        id=device_id,
        recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
        chronic=True,
    )

    # extract spiking activity
    # NOTE: this currently does not care about what brain region each spike is from
    spikes, units = extract_spikes_from_nwbfile(
        nwbfile, 
        recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
    )

    # filter out arrays that are not specified
    if args.array is not None:
        units = units.select_by_mask(units.array == args.array)
        spikes = spikes.select_by_mask(spikes.unit_array == ImplantArea.array_str_to_int[args.array])

        # Let remaining unit indices start from 0 consecutively
        # Step 1: Create the map (number -> "rank"/index)
        number_to_rank = {}
        for rank, number in enumerate(units.unit_number):
            number_to_rank[number] = rank
        
        # Step 2: Use the map to replace each element in data.spikes.unit_index
        for i, val in enumerate(spikes.unit_index):
            spikes.unit_index[i] = number_to_rank[val]

        # Step 3: Use the map to replace each element in data.units.unit_number
        for i, val in enumerate(units.unit_number):
            units.unit_number[i] = number_to_rank[val]

        if not len(units):
            raise ValueError(f"No units found for array {args.array}")

    # extract behavior
    cursor = extract_behavior(nwbfile)
    # cursor_outlier_segments = detect_outliers(cursor)

    # visualize_behavior(nwbfile, cursor)

    # extract data about trial structure
    trials, movement_phases = extract_trials(nwbfile, task, cursor)

    # close file
    io.close()

    # register session
    data = Data(
        brainset=brainset_description,
        subject=subject,
        session=session_description,
        device=device_description,
        # neural activity
        spikes=spikes,
        units=units,
        # stimuli and behavior
        trials=trials,
        movement_phases=movement_phases,
        cursor=cursor,
        # cursor_outlier_segments=cursor_outlier_segments,
        # domain
        domain=cursor.domain,
    )

    # split trials into train, validation and test
    successful_trials = trials.select_by_mask(trials.is_valid)
    _, valid_trials, test_trials = successful_trials.split(
        [0.7, 0.1, 0.2], shuffle=True, random_seed=42
    )

    train_sampling_intervals = data.domain.difference(
        (valid_trials | test_trials).dilate(3.0)
    )

    data.set_train_domain(train_sampling_intervals)
    data.set_valid_domain(valid_trials)
    data.set_test_domain(test_trials)

    # save data to disk
    path = os.path.join(args.output_dir, f"{session_id}.h5")
    with h5py.File(path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


if __name__ == "__main__":
    main()
