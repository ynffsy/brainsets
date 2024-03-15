"""Load data, processes it, save it."""

import argparse
import datetime
import logging
import os
import copy
import math
import torch
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import sys

sys.path.append("/home/mila/x/xuejing.pan/POYO/project-kirby")

from kirby.tasks.visual_coding import VISUAL_CODING
from kirby.data import Data, IrregularTimeSeries, Interval, DatasetBuilder, ArrayDict
from kirby.utils import find_files_by_extension
from kirby.taxonomy import *
from kirby.taxonomy.taxonomy import *

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from tqdm import tqdm

from data.scripts.openscope_calcium.utils import *


logging.basicConfig(level=logging.INFO)

# copying from neuropixel dataloader:
# https://github.com/nerdslab/project-kirby/blob/venky/allen/data/scripts/allen_visual_behavior_neuropixels/prepare_data.py
WINDOW_SIZE = 1.0
STEP_SIZE = 0.5
JITTER_PADDING = 0.25


def get_roi_position(ROI_masks, num_rois):
    """
    input: ROI object from allensdk

    output: sinusoidal encoding of all ROIs position
    """
    centroids = np.zeros((num_rois, 2))

    for count, curr in enumerate(ROI_masks):
        mask = ROI_masks[count].get_mask_plane()
        y_coords, x_coords = np.nonzero(mask)
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)

        centroids[count] = [centroid_y, centroid_x]

    sinu_pos = get_sinusoidal_encoding(centroids[:, 0], centroids[:, 1], num_dims=32)

    return sinu_pos


def get_roi_feats(ROI_masks, num_rois):
    """
    input: ROI object from allensdk

    output: minmax scaled area, height and width for each ROI
    """
    areas = np.zeros(num_rois)
    heights = np.zeros(num_rois)
    widths = np.zeros(num_rois)

    for count, curr in enumerate(ROI_masks):
        mask = ROI_masks[count].get_mask_plane()
        areas[count] = np.count_nonzero(mask)

        rows, cols = np.where(mask)
        heights[count] = np.max(rows) - np.min(rows) + 1
        widths[count] = np.max(cols) - np.min(cols) + 1

    normalized_areas = min_max_scale(areas, 101.0, 551.0)
    normalized_heights = min_max_scale(heights, 8.0, 44.0)
    normalized_widths = min_max_scale(widths, 9.0, 42.0)

    return normalized_areas, normalized_heights, normalized_widths


def get_stim_data(meta_table, timestamps):
    """
    inputs: stim table from allen sdk
            timestamps
    outputs:
    """
    stim_df = pd.DataFrame(meta_table)

    start_times = timestamps[stim_df.loc[(stim_df["blank_sweep"] == 0.0), "start"]]
    end_times = timestamps[stim_df.loc[(stim_df["blank_sweep"] == 0.0), "end"]]
    temp_freqs = stim_df.loc[(stim_df["blank_sweep"] == 0.0), "temporal_frequency"]
    orientations = stim_df.loc[(stim_df["blank_sweep"] == 0.0), "orientation"]

    trials = Interval(
        start=start_times,
        end=end_times,
        orientation=orientations.values.astype(np.float32),
        temporal_frequency=temp_freqs.values.astype(np.float32),
    )

    timestamps = np.concatenate([trials.start, trials.end])
    types = np.concatenate(
        [
            np.ones_like(trials.start) * VISUAL_CODING.STIMULUS_ON,
            np.ones_like(trials.end) * VISUAL_CODING.STIMULUS_OFF,
        ]
    )

    stimuli_events = IrregularTimeSeries(
        timestamps=timestamps,
        types=types,
    )

    ORIENTAIONS = [0, 45, 90, 135, 180, 225, 270, 315]
    stimuli_segments = copy.deepcopy(trials)
    stimuli_segments.drifting_class = np.round(trials.orientation / 45.0).astype(
        np.int64
    )
    stimuli_segments.drifting_temp_freq = trials.temporal_frequency.astype(np.int64)
    stimuli_segments.timestamps = np.ones_like(stimuli_segments.start) * 0.5

    return trials, stimuli_events, stimuli_segments


def main():
    # Use argparse to extract two arguments from the command line:
    # input_dir and output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    manifest_path = os.path.join(args.input_dir, "manifest.json")
    boc = BrainObservatoryCache(manifest_file=manifest_path)

    # Using a metadata file instead of dealing with filtering every single time
    meta_df = pd.read_csv(
        "/home/mila/x/xuejing.pan/POYO/project-kirby/data/scripts/allen_brain_observatory_calcium/AllenBOmeta.csv"
    )
    sess_ids = meta_df["exp_id"].values
    subject_ids = meta_df["subject_id"].values
    cre_lines = meta_df["cre_line"].values

    db = DatasetBuilder(
        raw_folder_path=args.input_dir,
        processed_folder_path=args.output_dir,
        # metadata for dataset
        experiment_name="allen_brain_observatory_calcium",
        origin_version="v2",
        derived_version="0.0.1",
        source="https://observatory.brain-map.org/visualcoding/",
        description="This dataset includes all experiments from "
        "Allen Institute Brain Observatory with stimulus drifting gratings.",
    )

    for count, curr_sess_id in enumerate(sess_ids):
        # FOR TESTING!!!!
        # if count >= 10:
        #    break
        with db.new_session() as session:
            print("AT FILE: ", count)

            nwbfile = boc.get_ophys_experiment_data(curr_sess_id)

            curr_meta_sess = nwbfile.get_metadata()

            cre_line_map = {
                "Cux2-CreERT2": Cre_line.CUX2_CREERT2,
                "Emx1-IRES-Cre": Cre_line.EXM1_IRES_CRE,
                "Fezf2-CreER": Cre_line.FEZF2_CREER,
                "Nr5a1-Cre": Cre_line.NR5A1_CRE,
                "Ntsr1-Cre_GN220": Cre_line.NTSR1_CRE_GN220,
                "Pvalb-IRES-Cre": Cre_line.PVALB_IRES_CRE,
                "Rbp4-Cre_KL100": Cre_line.RBP4_CRE_KL100,
                "Rorb-IRES2-Cre": Cre_line.RORB_IRES2_CRE,
                "Scnn1a-Tg3-Cre": Cre_line.SCNN1A_TG3_CRE,
                "Slc17a7-IRES2-Cre": Cre_line.SLC17A7_IRES2_CRE,
                "Sst-IRES-Cre": Cre_line.SST_IRES_CRE,
                "Tlx3-Cre_PL56": Cre_line.TLX3_CRE_PL56,
                "Vip-IRES-Cre": Cre_line.VIP_IRES_CRE,
            }
            sex_map = {"male": Sex.MALE, "female": Sex.FEMALE}

            subject = SubjectDescription(
                id=str(subject_ids[count]),
                species=Species.MUS_MUSCULUS,
                sex=sex_map[curr_meta_sess["sex"]],
                cre_line=cre_line_map[cre_lines[count]],
            )

            session.register_subject(subject)

            recording_date = curr_meta_sess["session_start_time"].strftime("%Y%m%d")
            sortset_id = curr_sess_id

            session.register_session(
                id=str(curr_sess_id),
                recording_date=recording_date,
                task=Task.DISCRETE_VISUAL_CODING,
                fields={
                    RecordingTech.OPENSCOPE_CALCIUM_TRACES: "spikes",
                    Output.DRIFTING_GRATINGS: "stimuli_segments.drifting_class",  # orientation
                    Output.DRIFTING_GRATINGS_TEMP_FREQ: "stimuli_segments.drifting_temp_freq",
                },
            )

            roi_ts, traces = nwbfile.get_dff_traces()
            traces_1d = torch.tensor(traces).view(-1)

            curr_num_rois = traces.shape[0]
            curr_num_frames = traces.shape[1]

            roi_ts_long = (torch.tensor(roi_ts)).repeat_interleave(curr_num_rois)
            roi_ids = nwbfile.get_roi_ids()

            traces = IrregularTimeSeries(
                timestamps=np.array((roi_ts_long)),
                unit_index=np.tile(
                    np.arange(0, curr_num_rois).astype(np.int16), (curr_num_frames,)
                ),
                values=np.array(traces_1d),
            )

            ROI_masks = nwbfile.get_roi_mask()
            unit_positions = get_roi_position(ROI_masks, curr_num_rois)
            unit_area, unit_width, unit_height = get_roi_feats(ROI_masks, curr_num_rois)

            units = ArrayDict(
                **{
                    "id": roi_ids,
                    "unit_names": roi_ids,
                    "unit_positions": np.array(unit_positions),
                    "unit_areas": np.array(unit_area),
                    "unit_widths": np.array(unit_width),
                    "unit_heights": np.array(unit_height),
                }
            )

            session.register_sortset(
                id=str(sortset_id),
                units=units,
            )

            master_stim_table = nwbfile.get_stimulus_table("drifting_gratings")
            trials, stimuli_events, stimuli_segments = get_stim_data(
                master_stim_table, roi_ts
            )

            stimulus_epochs = nwbfile.get_stimulus_epoch_table()
            session_start, session_end = (
                roi_ts[stimulus_epochs.iloc[0]["start"]],
                roi_ts[stimulus_epochs.iloc[-1]["end"]],
            )

            data = Data(
                # metadata
                start=session_start,
                end=session_end,
                session=curr_sess_id,
                sortset=sortset_id,
                subject=subject.id,
                # neural activity
                spikes=traces,
                units=units,
                # stimuli (and behaviour to come)
                trials=trials,
                stimuli_events=stimuli_events,
                stimuli_segments=stimuli_segments,
            )

            session.register_data(data)

            # split and register trials into train, validation and test
            train_trials, valid_trials, test_trials = trials.split(
                [0.7, 0.1, 0.2], shuffle=True, random_seed=42
            )

            session.register_split("train", train_trials)
            session.register_split("valid", valid_trials)
            session.register_split("test", test_trials)

            # trials.allow_split_mask_overlap()

            # save data to disk
            session.save_to_disk()

    # all sessions added, finish by generating a description file for the entire dataset
    db.finish()


if __name__ == "__main__":
    main()
