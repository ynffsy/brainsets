"""Load data, processes it, save it."""
import argparse
import datetime
import logging

import numpy as np
import torch
from pynwb import NWBHDF5IO
from scipy.ndimage import binary_dilation

from kirby.data import Data, IrregularTimeSeries, Interval, DatasetBuilder
from kirby.data.dandi_utils import (
    extract_spikes_from_nwbfile,
    extract_subject_from_nwb,
)
from kirby.tasks.reaching import REACHING
from kirby.utils import find_files_by_extension
from kirby.taxonomy import (
    Output,
    RecordingTech,
    Task,
)

logging.basicConfig(level=logging.INFO)


def extract_trials(nwbfile):
    r"""Extract trial information from the NWB file. Trials that are flagged as
    "to discard" or where the monkey failed are marked as invalid."""
    trial_table = nwbfile.trials.to_dataframe()

    # rename start and end time columns
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
            "target_presentation_time": "target_on_time",
        }
    )
    trials = Interval.from_dataframe(trial_table)

    is_valid = torch.logical_and(
        trials.discard_trial == 0.0, trials.task_success == 1.0
    )
    trials.is_valid = is_valid

    return trials


def extract_behavior(nwbfile, trials):
    """Extract behavior from the NWB file.

    ..note::
        Cursor position and target position are in the same frame of reference.
        They are both of size (sequence_len, 2). Finger position can be either 3d or 6d,
        depending on the sequence. # todo investigate more
    """
    # cursor, hand and eye share the same timestamps (verified)
    timestamps = nwbfile.processing["behavior"]["Position"]["Cursor"].timestamps[:]
    cursor_pos = nwbfile.processing["behavior"]["Position"]["Cursor"].data[:]  # 2d
    hand_pos = nwbfile.processing["behavior"]["Position"]["Hand"].data[:]
    eye_pos = nwbfile.processing["behavior"]["Position"]["Eye"].data[:]  # 2d

    # derive the velocity of the cursor
    cursor_vel = np.gradient(cursor_pos, timestamps, edge_order=1, axis=0)
    # derive the velocity and acceleration of the hand
    hand_vel = np.gradient(hand_pos, timestamps, edge_order=1, axis=0)
    hand_acc = np.gradient(hand_vel, timestamps, edge_order=1, axis=0)

    # normalization
    hand_vel = hand_vel / 800.0
    hand_acc = hand_acc / 800.0

    # create a behavior type segmentation mask
    timestamps = torch.tensor(timestamps)
    behavior_type = torch.ones_like(timestamps, dtype=torch.long) * REACHING.RANDOM
    for i in range(len(trials)):
        # first we check whether the trials are valid or not
        if trials.is_valid[i]:
            behavior_type[
                (timestamps >= trials.target_on_time[i])
                & (timestamps < trials.go_cue_time[i])
            ] = REACHING.CENTER_OUT_HOLD
            behavior_type[
                (timestamps >= trials.move_begins_time[i])
                & (timestamps < trials.move_ends_time[i])
            ] = REACHING.CENTER_OUT_REACH
            behavior_type[
                (timestamps >= trials.move_ends_time[i]) & (timestamps < trials.end[i])
            ] = REACHING.CENTER_OUT_RETURN

    # sometimes monkeys get angry, we want to identify the segments where the hand is
    # moving too fast, and mark them as outliers
    # we use the norm of the acceleration to identify outliers
    hand_acc_norm = np.linalg.norm(hand_acc, axis=1)
    mask = hand_acc_norm > 100.0
    # we dilate the mask to make sure we are not missing any outliers
    structure = np.ones(50, dtype=bool)
    mask = binary_dilation(mask, structure)
    behavior_type[mask] = REACHING.OUTLIER

    behavior = IrregularTimeSeries(
        timestamps=timestamps,
        cursor_pos=torch.tensor(cursor_pos),
        cursor_vel=torch.tensor(cursor_vel),
        hand_pos=torch.tensor(hand_pos),
        hand_vel=torch.tensor(hand_vel),
        hand_acc=torch.tensor(hand_acc),
        eye_pos=torch.tensor(eye_pos),
        type=behavior_type,
    )

    return behavior


def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    # intiantiate a DatasetBuilder which provides utilities for processing data
    db = DatasetBuilder(
        raw_folder_path=args.input_dir,
        processed_folder_path=args.output_dir,
        # metadata for the dataset
        experiment_name="churchland_shenoy_neural_2012",
        origin_version="dandi/000070/draft",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000070",
        description="Monkeys recordings of Motor Cortex (M1) and dorsal Premotor Cortex"
        " (PMd) using two 96 channel high density Utah Arrays (Blackrock Microsystems) "
        "while performing reaching tasks with right hand.",
    )

    # iterate over the .nwb files and extract the data from each
    for file_path in find_files_by_extension(db.raw_folder_path, ".nwb"):
        logging.info(f"Processing file: {file_path}")

        # each file contains data from one session. a session is unique and has one
        # associated subject and one associated sortset.
        with db.new_session() as session:
            # open file
            io = NWBHDF5IO(file_path, "r")
            nwbfile = io.read()

            # extract subject metadata
            # this dataset is from dandi, which has structured subject metadata, so we
            # can use the helper function extract_subject_from_nwb
            subject = extract_subject_from_nwb(nwbfile)
            session.register_subject(subject)

            # extract experiment metadata
            recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
            subject_id = subject.id
            sortset_id = f"{subject_id}_{recording_date}"
            session_id = f"{sortset_id}_center_out_reaching"

            # register session
            session.register_session(
                id=session_id,
                recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
                task=Task.DISCRETE_REACHING,
                fields={
                    RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS: "spikes",
                    RecordingTech.UTAH_ARRAY_WAVEFORMS: "spikes.waveforms",
                    Output.CURSOR2D: "behavior.cursor_vel",
                },
            )

            # extract spiking activity
            # this data is from dandi, we can use our helper function
            spikes, units = extract_spikes_from_nwbfile(
                nwbfile, recording_tech=RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS
            )

            # register sortset
            session.register_sortset(
                id=sortset_id,
                units=units,
            )

            # extract data about trial structure
            trials = extract_trials(nwbfile)

            # extract behavior
            behavior = extract_behavior(nwbfile, trials)

            # close file
            io.close()

            # register session
            session_start, session_end = (
                behavior.timestamps[0].item(),
                behavior.timestamps[-1].item(),
            )

            data = Data(
                # metadata
                start=session_start,
                end=session_end,
                session=f"{db.experiment_name}_{session_id}",
                sortset=f"{db.experiment_name}_{sortset_id}",
                subject=f"{db.experiment_name}_{subject_id}",
                # neural activity
                spikes=spikes,
                units=units,
                # stimuli and behavior
                trials=trials,
                behavior=behavior,
            )

            # split trials into train, validation and test
            train_trials, valid_trials, test_trials = trials[trials.is_valid].split(
                [0.7, 0.1, 0.2], shuffle=True, random_seed=42
            )

            # save samples
            session.register_samples_for_training(
                data, "train", exclude_intervals=[valid_trials, test_trials]
            )
            session.register_samples_for_evaluation(
                data, "valid", include_intervals=valid_trials
            )
            session.register_samples_for_evaluation(
                data, "test", include_intervals=test_trials
            )

            # save data to disk
            session.save_to_disk()

    # all sessions added, finish by generating a description file for the entire dataset
    db.finish()


if __name__ == "__main__":
    main()
