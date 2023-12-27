import collections
import datetime
import logging
import msgpack
import os
from pathlib import Path
import yaml

import numpy as np
import torch

from kirby.data import Data, Interval
from kirby.taxonomy import (
    ChunkDescription,
    DandisetDescription,
    SessionDescription,
    SortsetDescription,
    SubjectDescription,
    TrialDescription,
    to_serializable,
)
from kirby.utils import make_directory


class DatasetBuilder:
    def __init__(
        self,
        raw_folder_path,
        processed_folder_path,
        *,
        experiment_name,
        origin_version,
        derived_version,
        metadata_version="0.0.2",
        source,
        description,
        min_duration=1.0,
        window_size=1.0,
        stride=0.5,
        jitter=0.25,
    ):
        self.raw_folder_path = raw_folder_path
        self.processed_folder_path = processed_folder_path

        self.experiment_name = experiment_name
        self.origin_version = origin_version
        self.derived_version = derived_version
        self.metadata_version = metadata_version
        self.source = source
        self.description = description

        self.min_duration = min_duration
        self.window_size = window_size
        self.step_size = stride
        self.jitter = jitter

        # make processed folder if it doesn't exist
        # todo raise warning if it does exist, since some files might be overwritten
        make_directory(self.processed_folder_path, prompt_if_exists=False)
        make_directory(
            os.path.join(self.processed_folder_path, "train"), prompt_if_exists=False
        )
        make_directory(
            os.path.join(self.processed_folder_path, "valid"), prompt_if_exists=False
        )
        make_directory(
            os.path.join(self.processed_folder_path, "test"), prompt_if_exists=False
        )

        self.subjects = []
        self.sortsets = []

    def new_session(self):
        # initialize the session
        # each session should have 1 subject and 1 sortset
        return SessionContextManager(self)

    def is_subject_already_registered(self, subject_id):
        # register subject to the dandiset if it hasn't been registered yet
        return any([subject_id == member.id for member in self.subjects])

    def is_sortset_already_registered(self, sortset_id):
        # Check if the sortset is already registered
        return any([sortset_id == member.id for member in self.sortsets])

    def get_sortset(self, sortset_id):
        return next(
            (member for member in self.sortsets if member.id == sortset_id), None
        )

    def finish(self):
        # Transform sortsets to a list of lists, otherwise it won't serialize to yaml.
        description = DandisetDescription(
            id=self.experiment_name,
            origin_version=self.origin_version,
            derived_version=self.derived_version,
            metadata_version=self.metadata_version,
            source=self.source,
            description=self.description,
            folds=["train", "valid", "test"],
            subjects=self.subjects,
            sortsets=self.sortsets,
        )

        # Efficiently encode enums to strings
        description = to_serializable(description)

        filename = Path(self.processed_folder_path) / "description.yaml"
        print(f"Saving description to {filename}")

        with open(filename, "w") as f:
            yaml.dump(description, f)

        # For efficiency, we also save a msgpack version of the description.
        # Smaller on disk, faster to read.
        filename = Path(self.processed_folder_path) / "description.mpk"
        print(f"Saving description to {filename}")

        with open(filename, "wb") as f:
            msgpack.dump(description, f, default=encode_datetime)


class SessionContextManager:
    def __init__(self, builder):
        self.builder = builder

        self.chunks = collections.defaultdict(list)
        self.footprints = collections.defaultdict(list)

        self.subject = None
        self.sortset = None
        self.session = None

    def __enter__(self):
        return self

    def register_subject(self, subject: SubjectDescription = None, **kwargs):
        if self.subject is not None:
            raise ValueError(
                "A subject was already registered. A session can only have "
                "one subject."
            )

        # add subject to the session
        if subject is None:
            subject = SubjectDescription(**kwargs)
        self.subject = subject

        # if sortset was defined before subject, we need to update the reference
        if self.sortset is not None:
            self.sortset.subject = subject.id

    def register_sortset(
        self,
        sortset: SortsetDescription = None,
        *,
        id=None,
        units,
        sessions=[],
        areas=[],
        recording_tech=[],
        **kwargs,
    ):
        if self.sortset is not None:
            raise ValueError(
                "A sortset was already registered. A session can only have "
                "one sortset."
            )

        sortset_id = id if id is not None else self.sortset.id
        # add prefix to unit names
        units.unit_name = [
            f"{self.builder.experiment_name}/{sortset_id}/{unit}"
            for unit in units.unit_name
        ]

        if sortset is None:
            sortset = SortsetDescription(
                id=id,
                subject=self.subject.id if self.subject is not None else "",
                sessions=sessions,
                units=units.unit_name,
                areas=areas,
                recording_tech=recording_tech,
                **kwargs,
            )

        # Check if the sortset is already registered
        existing_sortset = self.builder.get_sortset(sortset.id)

        if existing_sortset is None:
            self.sortset = sortset
        else:
            # If it exists, update the reference
            self.sortset = existing_sortset

        # if session was defined before sortset, we need to update the reference
        if self.session is not None:
            self.sortset.sessions.append(self.session)

    def register_session(
        self, session: SessionDescription = None, *, trials=[], **kwargs
    ):
        if self.session is not None:
            raise ValueError(
                "A session description was already registered. A session "
                "can only have one description."
            )

        if session is None:
            session = SessionDescription(trials=trials, **kwargs)
        self.session = session

        # if sortset was defined before session, we need to update the reference
        if self.sortset is not None:
            self.sortset.sessions.append(session)

    def register_samples_for_training(
        self, data, fold, include_intervals=None, exclude_intervals=None
    ):
        assert (
            include_intervals is None or exclude_intervals is None
        ), "Cannot include and exclude intervals at the same time."

        if include_intervals is not None:
            raise NotImplementedError("Include intervals not implemented yet.")

        else:
            data_list = list(
                data.bucketize(
                    self.builder.window_size,
                    self.builder.step_size,
                    self.builder.jitter,
                )
            )
            if exclude_intervals is not None:
                if isinstance(exclude_intervals, Interval):
                    exclude_intervals = [exclude_intervals]
                for exclude_intervals_set in exclude_intervals:
                    data_list = self.exclude_intervals(data_list, exclude_intervals_set)

        self.save_to_disk(data_list, fold)

    def register_samples_for_evaluation(self, data, fold, include_intervals=None):
        if include_intervals is None:
            return
        data_list = self.slice_along_intervals(
            data, include_intervals, self.builder.min_duration
        )
        self.save_to_disk(data_list, fold)

    def save_to_disk(self, data_list, fold):
        for i, sample in enumerate(data_list):
            basename = f"{self.session.id}_{i:05}"
            filename = f"{basename}.pt"
            path = os.path.join(self.builder.processed_folder_path, fold, filename)
            torch.save(sample, path)

            self.footprints[fold].append(os.path.getsize(path))
            self.chunks[fold].append(
                ChunkDescription(
                    id=basename,
                    duration=(sample.end - sample.start).item(),
                    start_time=sample.start.item(),
                )
            )

    def exclude_intervals(self, data_list, exclude_intervals):
        out = []
        for i in range(len(data_list)):
            exclude = False
            bucket_start, bucket_end = data_list[i].start, data_list[i].end
            for interval in exclude_intervals:
                start, end = interval.start, interval.end
                if start <= bucket_end and end >= bucket_start:
                    exclude = True
                    break
            if not exclude:
                out.append(data_list[i])
        return out

    def slice_along_intervals(
        self, data: Data, intervals: Interval, min_duration: float
    ):
        data_list = []
        for interval in intervals:
            start, end = interval.start, interval.end
            if end - start <= min_duration:
                start = start - (min_duration - (end - start)) / 2
                end = start + min_duration
            data_list.append(data.slice(start, end))
        return data_list

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logging.error(f"Exception: {exc_type} {exc_value}")
            return False

        assert self.subject is not None, "A subject must be registered."
        assert self.sortset is not None, "A sortset must be registered."
        assert self.session is not None, "A session must be registered."

        self.footprints = {k: int(np.mean(v)) for k, v in self.footprints.items()}

        # add subject to the dandiset if it hasn't been registered yet
        if not self.builder.is_subject_already_registered(self.subject.id):
            self.builder.subjects.append(self.subject)

        # add sortset to the dandiset if it hasn't been registered yet
        if not self.builder.is_sortset_already_registered(self.sortset.id):
            self.builder.sortsets.append(self.sortset)

        # todo replace trial with epoch
        trial = TrialDescription(
            id=self.session.id,
            footprints=self.footprints,
            chunks=self.chunks,
        )

        self.session.trials = [trial]

        return True


def encode_datetime(obj):
    """msgpack doesn't support datetime, so we need to encode it as a string."""
    if isinstance(obj, datetime.datetime):
        return obj.strftime("%Y%m%dT%H:%M:%S.%f").encode()
