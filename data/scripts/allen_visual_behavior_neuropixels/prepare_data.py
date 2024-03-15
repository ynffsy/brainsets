"""Load data, processes it, delete un-needed attributes, save into sample chuncks."""

import argparse
import collections
import datetime
import logging
import os

import numpy as np
import pandas as pd
import torch
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from tqdm import tqdm

from kirby.data import ArrayDict, Data, Interval, IrregularTimeSeries
from kirby.tasks.visual_coding import VISUAL_CODING
from kirby.taxonomy import (
    ChunkDescription,
    DandisetDescription,
    DescriptionHelper,
    RecordingTech,
    SessionDescription,
    Sex,
    SortsetDescription,
    Species,
    Stimulus,
    SubjectDescription,
    Task,
    TrialDescription,
)
from kirby.utils import make_directory

logging.basicConfig(level=logging.INFO)


WINDOW_SIZE = 1.0
STEP_SIZE = 0.5
JITTER_PADDING = 0.25


def extract_spikes(units, prefix):
    units = session.units
    spiketimes_dict = session.spike_times

    spikes = []
    unit_index = []
    types = []
    # waveforms = []
    unit_meta = []

    for i, unit_id in enumerate(spiketimes_dict.keys()):
        metadata = units.loc[unit_id]
        probe_id = metadata["probe_id"]
        probe_channel_id = metadata["probe_channel_number"]
        unit_name = f"{prefix}/{probe_id}/{probe_channel_id}/{unit_id}"

        spiketimes = spiketimes_dict[unit_id]
        spikes.append(spiketimes)
        unit_index.append([i] * len(spiketimes))
        types.append(np.ones_like(spiketimes) * int(RecordingTech.NEUROPIXELS_SPIKES))

        unit_meta.append(
            {
                "count": len(spiketimes),
                "channel_name": probe_channel_id,
                "electrode_row": metadata["probe_horizontal_position"],
                "electrode_col": 0,
                "unit_name": unit_name,
                "area_name": metadata["structure_acronym"],
                "channel_number": probe_channel_id,
                "unit_number": i,
                "type": int(RecordingTech.NEUROPIXELS_SPIKES),
            }
        )

    spikes = np.concatenate(spikes)
    # waveforms = np.concatenate(waveforms)
    unit_index = np.concatenate(unit_index)
    types = np.concatenate(types)

    # convert unit metadata to a Data object
    unit_meta_df = pd.DataFrame(unit_meta)  # list of dicts to dataframe
    units = ArrayDict.from_dataframe(unit_meta_df, unsigned_to_long=True)

    sorted = np.argsort(spikes)
    spikes = spikes[sorted]
    # waveforms = waveforms[sorted]
    unit_index = unit_index[sorted]
    types = types[sorted]

    spikes = IrregularTimeSeries(
        timestamps=torch.tensor(spikes),
        # waveforms=torch.tensor(waveforms),
        unit_index=torch.tensor(unit_index),
        types=torch.tensor(types),
    )

    return spikes, units


def extract_behavior(session):
    # extract pupil size
    # extract running speed
    pass


def extract_trial_metadata(stimulus_pres):
    # to start, we will only consider the drifting gratings stimulus
    drifting_gratings = stimulus_pres[
        np.array(stimulus_pres["stimulus_name"] == "drifting_gratings")
        & np.array(stimulus_pres["orientation"] != "null")
    ]

    trials = Interval(
        start=torch.tensor(drifting_gratings["start_time"].values),
        end=torch.tensor(drifting_gratings["stop_time"].values),
        orientation=torch.tensor(
            drifting_gratings["orientation"].values.astype(np.float32)
        ),
        spatial_frequency=torch.tensor(
            drifting_gratings["spatial_frequency"].values.astype(np.float32)
        ),
        temporal_frequency=torch.tensor(
            drifting_gratings["temporal_frequency"].values.astype(np.float32)
        ),
    )

    stimuli_events = IrregularTimeSeries(
        timestamps=torch.cat([trials.start, trials.end]),
        type=torch.cat(
            [
                torch.ones_like(trials.start) * VISUAL_CODING.STIMULUS_ON,
                torch.ones_like(trials.end),
            ]
            * VISUAL_CODING.STIMULUS_OFF
        ),
    )

    REACH_DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]
    stimuli_segments = trials
    stimuli_segments.drifting_class = torch.round(trials.orientation / 45.0).long()
    # TODO for now, we will center all timestamps assuming a context window of 1s
    stimuli_segments.timestamps = torch.ones_like(stimuli_segments.start) * 0.5
    return trials, stimuli_events, stimuli_segments


def collect_slices(data, trials, min_duration=WINDOW_SIZE):
    slices = []
    for trial in trials:
        start, end = trial["start"], trial["end"]
        if end - start <= min_duration:
            start = start - (min_duration - (end - start)) / 2
            end = start + min_duration

        slices.append(data.slice(start, end))
    return slices


def encode_datetime(obj):
    """msgpack doesn't support datetime, so we need to encode it as a string."""
    if isinstance(obj, datetime.datetime):
        return obj.strftime("%Y%m%dT%H:%M:%S.%f").encode()


if __name__ == "__main__":
    experiment_name = "allen_visual_behavior_neuropixels_2019"

    # Use argparse to extract two arguments from the command line:
    # input_dir and output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()
    raw_folder_path = args.input_dir
    processed_folder_path = args.output_dir

    make_directory(processed_folder_path, prompt_if_exists=False)
    make_directory(os.path.join(processed_folder_path, "train"))
    make_directory(os.path.join(processed_folder_path, "valid"))
    make_directory(os.path.join(processed_folder_path, "test"))

    # Here, we will have multiple trials in each session
    helper = DescriptionHelper()
    session_list = []
    sortsets = []
    trials: list[TrialDescription] = []
    subjects = []

    manifest_path = os.path.join(args.input_dir, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    # get sessions
    sessions = cache.get_session_table()

    for session_id, row in tqdm(sessions.iterrows()):
        # load nwb file through the allen sdk
        session = cache.get_session_data(session_id)

        # new subject
        animal = f"mouse_{row['specimen_id']}"
        sex_map = {"M": Sex.MALE, "F": Sex.FEMALE}
        subjects.append(
            SubjectDescription(
                id=str(row["specimen_id"]),
                species=Species.MUS_MUSCULUS,
                age=row["age_in_days"],
                sex=sex_map[row["sex"]],
                genotype=row["full_genotype"],
            )
        )

        # there is only one session per subject.
        recording_date = session.session_start_time.strftime("%Y%m%d")
        sortset_id = f"{animal}_{recording_date}"

        # extract spiking activity
        spikes, units = extract_spikes(session, prefix=sortset_id)

        # extract session start and end times
        stimulus_epochs = session.get_stimulus_epochs()
        session_start, session_end = (
            stimulus_epochs.iloc[0]["start_time"],
            stimulus_epochs.iloc[-1]["stop_time"],
        )

        # extract behavior
        # behavior is None, for now
        behavior = extract_behavior(session)

        # extract trial structure
        stimulus_pres = session.stimulus_presentations  # trial table
        trials, stimuli_events, stimuli_segments = extract_trial_metadata(stimulus_pres)

        data = Data(
            start=session_start,
            end=session_end,
            spikes=spikes,
            units=units,
            behavior=behavior,
            trials=trials,
            stimuli_events=stimuli_events,
            stimuli_segments=stimuli_segments,
            session=str(session_id),
            sortset=sortset_id,
            subject=animal,
        )

        # get stimulus epoch blocks
        stimulus_epochs = stimulus_epochs[
            stimulus_epochs["stimulus_name"] == "drifting_gratings"
        ]

        if not (len(stimulus_epochs) == 3):
            logging.warning(
                f"There should be 3 stimulus epochs, found {len(stimulus_epochs)}."
            )
            continue

        train_slices = []
        valid_slices = []
        test_slices = []
        for i, (_, epoch) in enumerate(stimulus_epochs.iterrows()):
            epoch_start = epoch["start_time"]
            epoch_end = epoch["stop_time"]

            if i == 0:
                train_slices.append(data.slice(epoch_start, epoch_end))
            if i == 1:
                train_slices.append(
                    data.slice(epoch_start, 0.25 * epoch_start + 0.75 * epoch_end)
                )
                valid_slices.append(
                    data.slice(0.25 * epoch_start + 0.75 * epoch_end, epoch_end)
                )
            if i == 2:
                test_slices.append(data.slice(epoch_start, epoch_end))

        train_buckets = []
        valid_buckets = []
        test_buckets = []
        for slices, buckets in [
            (train_slices, train_buckets),
            (valid_slices, valid_buckets),
            (test_slices, test_buckets),
        ]:
            for slice in slices:
                for trial in slice.trials:
                    start, end = trial["start"], trial["end"]
                    if start > slice.end or end < slice.start:
                        continue
                    bucket_start = torch.tensor(max(slice.start, start - 0.5))
                    bucket_end = torch.tensor(min(slice.end, end + 0.5))
                    if bucket_end - bucket_start < 1.5:
                        logging.warning(
                            "Skipping bucket because it is too short: "
                            f"{bucket_start - bucket_end}"
                        )
                        continue
                    bucket = slice.slice(bucket_start, bucket_end)
                    buckets.append(bucket)

        chunks = collections.defaultdict(list)
        footprints = collections.defaultdict(list)

        logging.info(
            f"Found {len(train_buckets) + len(valid_buckets) + len(test_buckets)} total trials."
        )
        logging.info("Saving to disk.")
        for buckets, fold in [
            (train_buckets, "train"),
            (valid_buckets, "valid"),
            (test_buckets, "test"),
        ]:
            for i, sample in enumerate(buckets):
                basename = f"{session_id}_{i:05}"
                filename = f"{basename}.pt"
                path = os.path.join(processed_folder_path, fold, filename)
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

        # Create the metadata for description.mpk
        # Verify which areas are present in this sortset.
        areas = []  # todo

        sortset_description = SortsetDescription(
            id=sortset_id,
            subject=animal,
            areas=areas,
            recording_tech=[
                RecordingTech.NEUROPIXELS_SPIKES,
            ],
            sessions=[],
            units=data.units.unit_name.tolist(),
        )

        session = SessionDescription(
            id=str(session_id),
            recording_date=session.session_start_time,
            task=Task.DISCRETE_VISUAL_CODING,
            fields={
                RecordingTech.NEUROPIXELS_SPIKES: "spikes",
                Stimulus.DRIFTING_GRATINGS: "stimuli_segments.drifting_class",
            },
            trials=[],
        )

        trial = TrialDescription(
            id=f"{experiment_name}_{session_id}_01",
            chunks=chunks,
            footprints=footprints,
        )
        session.trials = [trial]

        helper.register_session(sortset_id, session)
        helper.register_sortset(experiment_name, sortset_description)

    # Create a description file for ease of reference.
    helper.register_dandiset(
        DandisetDescription(
            id="allen_visual_behavior_neuropixels_2019",
            origin_version="v2",  # allensdk version
            derived_version="0.0.1",  # This variant
            metadata_version="0.0.1",
            source="http://api.brain-map.org/",
            description="Visual Coding - Neuropixels from Allen Brain Observatory (2019).",
            folds=["train", "valid", "test"],
            subjects=subjects,
            sortsets=[],
        )
    )

    description = helper.finalize()
    helper.write_to_disk(processed_folder_path, description)
