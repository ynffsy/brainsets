"""Load data, processes it, delete un-needed attributes, save into sample chuncks."""
import argparse
import collections
import datetime
import logging
import os
from pathlib import Path
from typing import List, Tuple

import h5py
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
    signal,
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
SAMPLE_FREQUENCY = 24414.0625


def generate_sortset_description(
    id: str,
    subject_name: str,
    areas: list[StringIntEnum],
    broadband: bool,
) -> SortsetDescription:
    """Generate sortset information."""
    recording_tech = [
        RecordingTech.UTAH_ARRAY_SPIKES,
        RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS,
        RecordingTech.UTAH_ARRAY_WAVEFORMS,
        RecordingTech.UTAH_ARRAY_AVERAGE_WAVEFORMS,
    ]

    if broadband:
        recording_tech.append(RecordingTech.UTAH_ARRAY_LFPS)

    return SortsetDescription(
        id=id,
        subject=subject_name,
        areas=areas,
        recording_tech=recording_tech,
        sessions=[],
        units=[],
    )


def generate_session_description(
    id: str,
    duration: float,
    recording_date: str,
    broadband: bool,
) -> SessionDescription:
    """Generate trial and session information.

    Here, we have one trial = one session. This is often the case in continuous
    behavioral paradigms, but not in discrete ones.
    """
    inputs = {
        RecordingTech.UTAH_ARRAY_SPIKES: "spikes",
        RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS: "spikes",
        RecordingTech.UTAH_ARRAY_WAVEFORMS: "spikes.waveforms",
        RecordingTech.UTAH_ARRAY_AVERAGE_WAVEFORMS: "units.average_waveform"
    }

    if broadband:
        inputs[RecordingTech.UTAH_ARRAY_LFPS] = "lfps"

    return SessionDescription(
        id=id,
        start_time=datetime.datetime.strptime(recording_date, "%Y%m%d"),
        end_time=datetime.datetime.strptime(recording_date, "%Y%m%d")
        + datetime.timedelta(seconds=duration),
        task=Task.CONTINUOUS_REACHING,
        inputs=inputs,
        stimuli={Stimulus.TARGET2D: "behavior.target_pos"},
        outputs={
            Output.CURSOR2D: "behavior.cursor_pos",
            Output.FINGER3D: "behavior.finger_vel",
            Output.CONTINUOUS_TRIAL_ONSET_OFFSET: "behavior.trial_onset_offset",
        },
        trials=[],
    )


# We could read this from one of the LFP hdf5 file, but it's small enough that we can
# just hardcode it.
channel_map = np.array(
    [
        [0, 0, -1000],
        [0, 2000, -1000],
        [1200, 800, -1000],
        [800, 1600, -1000],
        [400, 400, -1000],
        [-400, 2000, -1000],
        [800, 1200, -1000],
        [400, 2000, -1000],
        [0, 400, -1000],
        [2400, -1200, -1000],
        [1200, 1200, -1000],
        [2400, -800, -1000],
        [0, 800, -1000],
        [2000, -1200, -1000],
        [400, 800, -1000],
        [2000, -800, -1000],
        [0, 1200, -1000],
        [1600, -1200, -1000],
        [400, 1200, -1000],
        [1600, -800, -1000],
        [0, 1600, -1000],
        [1200, -1200, -1000],
        [400, 1600, -1000],
        [1200, -800, -1000],
        [-800, 0, -1000],
        [800, -1200, -1000],
        [-400, 0, -1000],
        [800, -800, -1000],
        [-400, 400, -1000],
        [400, -1200, -1000],
        [-800, 400, -1000],
        [400, -800, -1000],
        [-400, 800, -1000],
        [0, -1200, -1000],
        [-800, 800, -1000],
        [0, -800, -1000],
        [-400, 1200, -1000],
        [-400, -1200, -1000],
        [-800, 1200, -1000],
        [-400, -800, -1000],
        [-800, 1600, -1000],
        [-800, -800, -1000],
        [-400, 1600, -1000],
        [0, -400, -1000],
        [-400, 2400, -1000],
        [-800, -400, -1000],
        [-800, 2000, -1000],
        [-400, -400, -1000],
        [2800, -400, -1000],
        [2000, 1200, -1000],
        [2800, -800, -1000],
        [2800, 2000, -1000],
        [2800, 0, -1000],
        [1600, 800, -1000],
        [2800, 400, -1000],
        [2400, 2000, -1000],
        [2000, 400, -1000],
        [2000, 800, -1000],
        [2800, 800, -1000],
        [2400, 2400, -1000],
        [2400, 400, -1000],
        [1600, 1200, -1000],
        [2800, 1200, -1000],
        [2000, 2000, -1000],
        [2400, 800, -1000],
        [1200, 1600, -1000],
        [2800, 1600, -1000],
        [2000, 2400, -1000],
        [2400, 1200, -1000],
        [2000, 1600, -1000],
        [2400, 1600, -1000],
        [1600, 2400, -1000],
        [1600, -400, -1000],
        [1600, 1600, -1000],
        [1600, 0, -1000],
        [1200, 2400, -1000],
        [1200, -400, -1000],
        [1600, 2000, -1000],
        [1200, 0, -1000],
        [800, 2400, -1000],
        [800, -400, -1000],
        [1200, 2000, -1000],
        [1200, 400, -1000],
        [400, 2400, -1000],
        [800, 0, -1000],
        [800, 2000, -1000],
        [800, 400, -1000],
        [0, 2400, -1000],
        [400, -400, -1000],
        [2400, -400, -1000],
        [800, 800, -1000],
        [2400, 0, -1000],
        [400, 0, -1000],
        [2000, -400, -1000],
        [1600, 400, -1000],
        [2000, 0, -1000],
    ]
)


def generate_probe_description() -> list[Probe]:
    # In this case, there are exactly 4 probes, 2 for each animal. We have the exact
    # locations for the probes that have local field potential info, (indy m1), but
    # not for the ones that don't.
    infos = [
        ("indy_m1", Macaque.primary_motor_cortex, "M1"),
        ("indy_s1", Macaque.primary_somatosensory_cortex, "S1"),
        ("loco_m1", Macaque.primary_motor_cortex, "M1"),
        ("loco_s1", Macaque.primary_somatosensory_cortex, "S1"),
    ]

    descriptions = []
    for suffix, area, name in infos:
        channels = [
            Channel(
                id=f"{name} {i+1:03}",
                local_index=i,
                relative_x_um=channel_map[i, 0] if "suffix" == "indy_m1" else 0,
                relative_y_um=channel_map[i, 1] if "suffix" == "indy_m1" else 0,
                relative_z_um=channel_map[i, 2] if "suffix" == "indy_m1" else 0,
                area=area,
            )
            for i in range(96)
        ]

        description = Probe(
            id=f"odoherty_sabes_{suffix}",
            type=RecordingTech.UTAH_ARRAY,
            wideband_sampling_rate=SAMPLE_FREQUENCY,
            waveform_sampling_rate=SAMPLE_FREQUENCY,
            lfp_sampling_rate=500,
            waveform_samples=48,
            channels=channels,
        )
        descriptions.append(description)

    return descriptions


def encode_datetime(obj):
    """msgpack doesn't support datetime, so we need to encode it as a string."""
    if isinstance(obj, datetime.datetime):
        return obj.strftime("%Y%m%dT%H:%M:%S.%f").encode()


def identify_outliers(data, threshold=6000):
    hand_acc_norm = np.linalg.norm(data.behavior.cursor_acc, axis=1)
    mask = hand_acc_norm > threshold
    structure = np.ones(100, dtype=bool)
    # Dilate the binary mask
    dilated = binary_dilation(mask, structure)
    return dilated


def extract_behavior(h5file):
    """Extract the behavior from the h5 file.

    ..note::
        Cursor position and target position are in the same frame of reference. They are both of size (sequence_len, 2).
        Finger position can be either 3d or 6d, depending on the sequence. # todo investigate more
    """
    cursor_pos = h5file["cursor_pos"][:].T
    finger_pos = h5file["finger_pos"][:].T
    target_pos = h5file["target_pos"][:].T
    timestamps = h5file["t"][:][0]

    # calculate the velocity of the cursor
    cursor_vel = np.gradient(cursor_pos, timestamps, edge_order=1, axis=0)
    cursor_acc = np.gradient(cursor_vel, timestamps, edge_order=1, axis=0)
    finger_vel = np.gradient(finger_pos, timestamps, edge_order=1, axis=0)

    # TODO: Refactor this for reusability.
    # Extract two traces that capture the target and movement onsets.
    # Similar to https://www.biorxiv.org/content/10.1101/2021.11.21.469441v3.full.pdf
    cursor_vel = np.gradient(cursor_pos, timestamps, edge_order=1, axis=0)
    cursor_vel_abs = np.sqrt(cursor_vel[:, 0] ** 2 + cursor_vel[:, 1] ** 2)

    # Dirac delta whenever the target changes.
    delta_time = np.sqrt((np.diff(target_pos, axis=0) ** 2).sum(axis=1)) >= 1e-9
    delta_time = np.concatenate([delta_time, [0]])

    tics = np.where(delta_time)[0]

    thresh = 0.2

    # Find the maximum for each integer value of period.
    max_times = np.zeros(len(tics) - 1, dtype=int)
    reaction_times = np.zeros(len(tics) - 1, dtype=int)
    for i in range(len(tics) - 1):
        max_vel = cursor_vel_abs[tics[i] : tics[i + 1]].max()
        reaction_times[i] = np.where(
            cursor_vel_abs[tics[i] : tics[i + 1]] >= thresh * max_vel
        )[0][0]
        max_times[i] = reaction_times[i] + tics[i]

    # Transform it back to a Dirac delta.
    start_times = np.zeros_like(delta_time)
    start_times[max_times] = 1

    behavior = IrregularTimeSeries(
        timestamps=torch.tensor(timestamps),
        cursor_pos=torch.tensor(cursor_pos),
        cursor_vel=torch.tensor(cursor_vel),
        hand_vel=torch.tensor(cursor_vel)
        / 200.0,  # todo: this is used to match the other datasets
        cursor_acc=torch.tensor(cursor_acc),
        type=torch.ones(len(timestamps), dtype=torch.int64) * REACHING.RANDOM,
        target_pos=torch.tensor(target_pos),
        finger_pos=torch.tensor(finger_pos),
        finger_vel=torch.tensor(finger_vel),
        trial_onset_offset=torch.stack(
            [torch.tensor(start_times), torch.tensor(delta_time)], dim=1
        ),
    )
    return behavior


def extract_lfp(
    h5file: h5py.File, channels: List[str]
) -> Tuple[RegularTimeSeries, Data]:
    """Extract the LFP from the h5 file."""
    logging.info("Broadband data attached. Computing LFP.")
    timestamps = h5file.get("/acquisition/timeseries/broadband/timestamps")[:].squeeze()

    # unfortunately, we have to chunk this because it's too big to fit in memory.
    n_samples_per_chunk = int(SAMPLE_FREQUENCY * 128)
    assert n_samples_per_chunk % 1000 == 0
    n_chunks = int(
        np.ceil(
            h5file.get("/acquisition/timeseries/broadband/data").shape[0]
            / n_samples_per_chunk
        )
    )

    lfps = []
    t_lfps = []
    for i in tqdm(range(n_chunks)):
        # Slow, iterative algorithm to prevent OOM issues.
        # Easiest would be to do this by channel but the memory layout in hdf5 doesn't
        # permit doing this efficiently.
        rg = slice(i * n_samples_per_chunk, (i + 1) * n_samples_per_chunk)
        broadband = h5file.get("/acquisition/timeseries/broadband/data")[rg, :]
        lfp, t_lfp = signal.downsample_wideband(
            broadband, timestamps[rg], SAMPLE_FREQUENCY
        )

        lfps.append(lfp.squeeze())
        t_lfps.append(t_lfp)

    lfp = np.concatenate(lfps, axis=0)
    t_lfp = np.concatenate(t_lfps)

    assert lfp.shape[0] == t_lfp.shape[0]
    assert lfp.ndim == 2
    assert t_lfp.ndim == 1

    lfp_bands, t_lfp_bands, names = signal.extract_bands(lfp, t_lfp)
    lfp = RegularTimeSeries(
        timestamps=torch.tensor(t_lfp_bands),
        lfp=torch.tensor(lfp_bands),
    )

    lfp_metadata = Data(channels=channels, bands=names)

    return lfp, lfp_metadata


def load_references_2d(h5file, ref_name):
    return [[h5file[ref] for ref in ref_row] for ref_row in h5file[ref_name][:]]


def to_ascii(vector):
    return ["".join(chr(char) for char in row) for row in vector]


def extract_spikes(h5file: h5py.File, prefix: str):
    r"""This dataset has a mixture of sorted and unsorted (threshold crossings) units."""
    spikesvec = load_references_2d(h5file, "spikes")
    waveformsvec = load_references_2d(h5file, "wf")

    # This is slightly silly but we can convert channel names back to an ascii token this way.
    chan_names = to_ascii(np.array(load_references_2d(h5file, "chan_names")).squeeze())

    spikes = []
    names = []
    types = []
    waveforms = []
    unit_meta = []

    # The 0'th spikesvec corresponds to unsorted thresholded units, the rest are sorted.
    suffixes = ["unsorted"] + [f"sorted_{i:02}" for i in range(1, 11)]
    type_map = [int(RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS)] + [
        int(RecordingTech.UTAH_ARRAY_SPIKES)
    ] * 10

    # Map from common names to brodmann areas
    bas = {"s1": 3, "m1": 4}

    for j in range(len(spikesvec)):
        crossings = spikesvec[j]
        for i in range(len(crossings)):
            spiketimes = crossings[i][:][0]
            if spiketimes.ndim == 0:
                continue

            spikes.append(spiketimes)
            area, channel_number = chan_names[i].split(" ")

            unit_name = f"{prefix}/{chan_names[i]}/{suffixes[j]}"
            names.append([unit_name] * len(spiketimes))
            types.append(np.ones_like(spiketimes, dtype=np.int64) * type_map[j])

            wf = np.array(waveformsvec[j][i][:])
            unit_meta.append(
                {
                    "count": len(spiketimes),
                    "channel_name": chan_names[i],
                    "unit_name": unit_name,
                    "area_name": area,
                    "channel_number": channel_number,
                    "unit_number": j,
                    "ba": bas[area.lower()],
                    "type": type_map[j],
                    "average_waveform": wf.mean(axis=1),
                    # Based on https://zenodo.org/record/1488441
                    "waveform_sampling_rate": 24414.0625,
                }
            )
            waveforms.append(wf.T)

    spikes = np.concatenate(spikes)
    waveforms = np.concatenate(waveforms)
    names = np.concatenate(names)
    types = np.concatenate(types)

    # Cast to torch tensors
    unit_meta_long = {}
    for key, item in unit_meta[0].items():
        if np.issubdtype(type(item), np.number):
            unit_meta_long[key] = torch.tensor(
                np.stack([x[key] for x in unit_meta], axis=0)
            )
        else:
            unit_meta_long[key] = np.stack([x[key] for x in unit_meta], axis=0)

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
    return spikes, units, chan_names


def split_and_get_train_valid_test(start, end, random_state=42):
    r"""
    From the paper: We split each recording session into 10 nonoverlapping contiguous segments of equal size which
    were then categorised into three different sets: training set (8 concatenated segments), valid set (1
    segment) and testing set (1 segment)."""
    intervals = np.linspace(start, end, 11)
    start, end = intervals[:-1], intervals[1:]

    segment_ids = np.arange(10)
    rng = np.random.default_rng(random_state)
    segments_ids = rng.permutation(segment_ids)

    train_segments = list(zip(start[segments_ids[:8]], end[segments_ids[:8]]))
    valid_segments = [(start[segments_ids[8]], end[segments_ids[8]])]
    test_segments = [(start[segments_ids[9]], end[segments_ids[9]])]

    return train_segments, valid_segments, test_segments


def collect_slices(data, segments):
    slices = []
    for start, end in segments:
        slices.append(data.slice(start, end))
    return slices


if __name__ == "__main__":
    experiment_name = "odoherty_sabes_reaching_2017"

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

    extension = ".mat"
    session_list = []

    # Here, we have one trial = one session. This is often the case in continuous
    # behavioral paradigms, but not in discrete ones.
    sortsets = {}
    trials: list[TrialDescription] = []

    # We don't have any info about age of sex for these subjects.
    subjects = [
        SubjectDescription(id="indy", species=Species.MACACA_MULATTA),
        SubjectDescription(id="loco", species=Species.MACACA_MULATTA),
    ]

    probes = generate_probe_description()

    # find all files with extension .mat in folder_path
    for file_path in tqdm(sorted(find_files_by_extension(raw_folder_path, extension))):
        # extract spikes
        logging.info(f"Processing file: {file_path}")
        session_id = Path(file_path).stem  # type: ignore

        sortset_id = session_id[:-3]
        assert sortset_id.count("_") == 1, f"Unexpected file name: {sortset_id}"
        animal, recording_date = sortset_id.split("_")

        broadband_path = Path(raw_folder_path) / "broadband" / f"{session_id}.nwb"
        # Check if the broadband data file exists.
        broadband = broadband_path.exists()

        h5file = h5py.File(file_path, "r")

        # extract behavior
        behavior = extract_behavior(h5file)
        start, end = behavior.timestamps[0].item(), behavior.timestamps[-1].item()
        spikes, units, chan_names = extract_spikes(h5file, sortset_id)

        # Extract LFPs
        extras = dict()
        if broadband:
            # Load the associated broadband data.
            broadband_file = h5py.File(broadband_path, "r")
            extras["lfps"], extras["lfp_metadata"] = extract_lfp(
                broadband_file, chan_names
            )

        # Assemble probe information. It's a bit awkward because we have the info in the
        # case of local field potentials, but not in the case of spikes.
        relevant_probe_names = set(
            [f"odoherty_sabes_{animal}_{x[:2]}".lower() for x in chan_names]
        )
        relevant_probes = []

        for probe in probes:
            if probe.id in relevant_probe_names:
                relevant_probes.append(probe)

        if len(relevant_probes) == 0:
            raise ValueError(f"No probes found for {sortset_id}")

        data = Data(
            spikes=spikes,
            units=units,
            behavior=behavior,
            start=start,
            end=end,
            probes=relevant_probes,
            # These are all the string metadata that we have. Later, we'll use this for
            # keying into EmbeddingWithVocab embeddings.
            session=f"{experiment_name}_{session_id}",
            sortset=f"{experiment_name}_{sortset_id}",
            subject=f"{experiment_name}_{animal}",
            **extras,
        )

        mask = identify_outliers(data)
        data.behavior.type[mask] = REACHING.OUTLIER

        # get successful trials, and keep 20% for test, 10% for valid
        (
            train_segments,
            valid_segments,
            test_segments,
        ) = split_and_get_train_valid_test(start, end)

        # collect data slices for valid and test trials
        train_slices = collect_slices(data, train_segments)
        valid_slices = collect_slices(data, valid_segments)
        test_slices = collect_slices(data, test_segments)

        # the remaining data (unstructured) is used for training
        train_buckets = []
        for segment in train_slices:
            # Note that we keep the original timestamps to facilitate soft/hard example
            # mining in contrastive learning.
            old_segment_start = segment.start
            segment.start, segment.end = 0, segment.end - segment.start

            # When we sliced the data, this subtracted the start from the timestamps. We
            # need to add it back to get the correct timestamps. This is important for
            # soft/hard example mining.
            buckets = list(segment.bucketize(WINDOW_SIZE, STEP_SIZE, JITTER_PADDING))
            for bucket in buckets:
                bucket.start += old_segment_start
                bucket.end += old_segment_start

            train_buckets.extend(buckets)

        chunks = collections.defaultdict(list)
        footprints = collections.defaultdict(list)

        logging.info("Saving to disk.")
        for buckets, fold in [
            (train_buckets, "train"),
            (valid_slices, "valid"),
            (test_slices, "test"),
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

        # Create the metadata for description.yaml
        # Get the channel names from the regular file.
        area_map = {
            "m1": Macaque.primary_motor_cortex,
            "s1": Macaque.primary_somatosensory_cortex,
        }
        if sortset_id not in sortsets:
            # Verify which areas are present in this sortset.
            areas = set([x.split(" ")[0].lower() for x in chan_names])
            areas = [area_map[x] for x in areas]
            sortsets[sortset_id] = generate_sortset_description(
                sortset_id, animal, areas, broadband
            )

        session = generate_session_description(
            f"{experiment_name}_{session_id}", end - start, recording_date, broadband
        )

        trial = TrialDescription(
            id=f"{experiment_name}_{session_id}_01",
            chunks=chunks,
            footprints=footprints,
        )
        session.trials = [trial]

        sortsets[sortset_id].units.append(units.unit_name)
        sortsets[sortset_id].sessions.append(session)

        h5file.close()

    # Transform sortsets to a list of lists, otherwise it won't serialize to yaml.
    sortsets = sorted(list(sortsets.values()), key=lambda x: x.id)
    for x in sortsets:
        x.units = sorted(list(set(np.concatenate(x.units).tolist())))

    # Create a description file for ease of reference.
    description = DandisetDescription(
        id="odoherty_sabes_reaching_2017",
        origin_version="583331",  # Zenodo version
        derived_version="0.0.1",  # This variant
        metadata_version="0.0.1",
        source="https://zenodo.org/record/583331",
        description="Reaching dataset from O'Doherty et al. (2017), data from M1 and S1.",
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
