import argparse
import collections
from pathlib import Path
from typing import List

import dateutil
import numpy as np
import torch
from scipy.io import loadmat

from kirby.data import signal
from kirby.data.data import (
    Channel,
    Data,
    Hemisphere,
    IrregularTimeSeries,
    Probe,
)
from kirby.taxonomy import Output, SessionDescription, Task, writing
from kirby.taxonomy.description_helper import DescriptionHelper
from kirby.taxonomy.macaque import Macaque
from kirby.taxonomy.taxonomy import (
    ChunkDescription,
    DandisetDescription,
    RecordingTech,
    SortsetDescription,
    Species,
    SubjectDescription,
    TrialDescription,
)

experiment_name = "willett_shenoy"
subject_name = f"{experiment_name}_t5"


def generate_probe_description() -> List[Probe]:
    probes = []
    for location in ["lateral", "medial"]:
        channels = []
        for i in range(96):
            channels = [
                Channel(
                    f"{subject_name}_{location}_{i:03}",
                    i,
                    0,
                    0,
                    0,
                    Macaque.primary_motor_cortex,  # This is human but using Macaque nomenclature for now.
                    Hemisphere.LEFT,
                )
            ]

        probes.append(
            Probe(
                f"{subject_name}_{location}",
                RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS,
                0,
                0,
                0,
                0,
                channels=channels,
            )
        )
    return probes


def process_single_letters(
    session_path: Path,
    processed_folder_path: Path,
    straight_lines: bool = False,
):
    sortset_name = f"{subject_name}/{session_path.parent.parts[-1]}"
    probes = generate_probe_description()
    if straight_lines:
        session_name = f"{sortset_name}_straight_lines"
    else:
        session_name = f"{sortset_name}_single_letters"
    single_letters_data = loadmat(session_path)

    labels = []
    spike_cubes = []
    train_masks = []
    valid_masks = []
    test_masks = []

    for key in single_letters_data.keys():
        if not key.startswith("neuralActivityCube_"):
            continue

        letter = key[len("neuralActivityCube_") :]
        resolved = False
        found = False
        try:
            resolved = writing.Character[letter]
            found = True
        except:
            pass

        try:
            resolved = writing.Line[letter]
            found = True
        except:
            pass

        assert found

        data = single_letters_data[f"neuralActivityCube_{letter}"]
        if data.min() > 1:
            continue

        spike_cubes.append(data)
        labels += [int(resolved)] * len(spike_cubes)
        valid_mask = np.arange(len(spike_cubes)) % 9 == 2
        test_mask = np.arange(len(spike_cubes)) % 9 == 5
        train_mask = ~valid_mask & ~test_mask
        train_masks.append(train_mask)
        valid_masks.append(valid_mask)
        test_masks.append(test_mask)

    train_masks = np.concatenate(train_masks)
    valid_masks = np.concatenate(valid_masks)
    test_masks = np.concatenate(test_masks)

    folds = np.where(
        train_masks, "train", np.where(valid_masks, "valid", "test")
    )

    spike_cubes = np.concatenate(spike_cubes, axis=0)

    # We only select 1 second, which was the length of the go period.
    spike_cubes = spike_cubes[:, 51:151, :]
    ts = (
        np.arange(0.5, 0.5 + spike_cubes.shape[1]) / 100.0
    )  # 100 Hz sampling rate

    channel_prefix = f"{sortset_name}/channel_"
    trials, units = signal.cube_to_long(
        ts, spike_cubes, channel_prefix=channel_prefix
    )
    labels = np.array(labels)

    # TODO: use the geometry map.
    counters = collections.defaultdict(int)
    trial_descriptions = []

    for trial, label, fold in zip(trials, labels, folds):
        behavior = IrregularTimeSeries(
            timestamps=torch.Tensor([0]), 
            letters=torch.tensor([[int(label)]]),
            behavior_type=torch.tensor([0])
        )
        data = Data(
            spikes=trial,
            units=units,
            behavior=behavior,
            start=0,
            end=1.0,
            probes=probes,
            session=session_name,
            sortset=sortset_name,
            subject=subject_name,
        )
        i = counters[fold]
        basename = f"{session_name}_{i:05}"
        filename = f"{basename}.pt"
        path = processed_folder_path / fold / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(data, path)

        chunk_description = ChunkDescription(
            id=basename,
            duration=1,
            start_time=0,  # Not clear from the blocks here, unfortunately.
        )

        trial_descriptions.append(
            TrialDescription(
                id=basename, footprints={}, chunks={fold.item(): [chunk_description]}
            )
        )

        counters[fold] += 1

    if straight_lines:
        task = Task.DISCRETE_WRITING_LINE
        output = Output.WRITING_LINE
    else:
        task = Task.DISCRETE_WRITING_CHARACTER
        output = Output.WRITING_CHARACTER

    session = SessionDescription(
        id=session_name,
        start_time=dateutil.parser.parse(
            single_letters_data.get("blockStartDates")[0][0].item()
        ),
        end_time=dateutil.parser.parse(
            single_letters_data.get("blockStartDates")[-1][-1].item()
        ),
        task=task,
        inputs={RecordingTech.UTAH_ARRAY: "spikes"},
        stimuli={},
        outputs={output: "behavior.letters"},
        trials=trial_descriptions,
    )

    return sortset_name, units.unit_name, session


# Load the straightLines.mat file
def load_straight_lines(session_path):
    straight_lines_data = loadmat(session_path + "straightLines.mat")

    # Spikes data structure to hold neuralActivityCube
    spikes = {}
    spikes["neural_activity_cubes"] = straight_lines_data.get(
        "neuralActivityCube_{x}"
    )
    spikes["neural_activity_time_series"] = straight_lines_data.get(
        "neuralActivityTimeSeries"
    )

    # Behaviour data structure to hold different behavioral aspects
    behaviour = {}
    behaviour["clock_time_series"] = straight_lines_data.get("clockTimeSeries")
    behaviour["block_nums_time_series"] = straight_lines_data.get(
        "blockNumsTimeSeries"
    )
    behaviour["go_cue_onset_time_bin"] = straight_lines_data.get(
        "goCueOnsetTimeBin"
    )
    behaviour["delay_cue_onset_time_bin"] = straight_lines_data.get(
        "delayCueOnsetTimeBin"
    )

    # Outputs data structure to hold output details
    outputs = {}
    outputs["means_per_block"] = straight_lines_data.get("meansPerBlock")
    outputs["std_across_all_data"] = straight_lines_data.get(
        "stdAcrossAllData"
    )
    outputs["array_geometry_map"] = straight_lines_data.get("arrayGeometryMap")

    return spikes, behaviour, outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()
    raw_folder_path = args.input_dir
    processed_folder_path = args.output_dir

    partitions = loadmat(
        Path(raw_folder_path)
        / "RNNTrainingSteps"
        / "trainTestPartitions_HeldOutBlocks.mat"
    )
    helper = DescriptionHelper()

    files = sorted(
        list((Path(raw_folder_path) / "Datasets").glob("*/singleLetters.mat"))
        + list(
            (Path(raw_folder_path) / "Datasets").glob("*/straightLines.mat")
        )
    )

    for file in files:
        sortset_name, channel_names, session = process_single_letters(
            file,
            Path(processed_folder_path),
            "straightLines.mat" in file.parts,
        )

        sortset_description = SortsetDescription(
            id=sortset_name,
            subject=subject_name,
            areas=[Macaque.primary_motor_cortex],
            recording_tech=[RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS],
            sessions=[],
            units=channel_names,
        )

        helper.register_session(sortset_name, session)
        helper.register_sortset(experiment_name, sortset_description)

    helper.register_dandiset(
        DandisetDescription(
            id=experiment_name,
            origin_version="0.0.0",
            derived_version="0.0.0",
            metadata_version="0.0.0",
            source="https://datadryad.org/stash/dataset/doi:10.5061/dryad.wh70rxwmv",
            description="Handwriting BCI data from Willett and Shenoy",
            folds=["train", "valid", "test"],
            subjects=[
                SubjectDescription(
                    id=subject_name,
                    species=Species.HOMO_SAPIENS,
                )
            ],
            sortsets=[],
        )
    )

    description = helper.finalize()
    helper.write_to_disk(processed_folder_path, description)