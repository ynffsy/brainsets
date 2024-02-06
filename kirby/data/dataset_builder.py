import datetime
import logging
import msgpack
import os
from pathlib import Path
import yaml
from typing import (
    List,
    Optional,
    Union,
    Tuple,
)

import h5py
import numpy as np

from kirby.taxonomy import (
    DandisetDescription,
    SessionDescription,
    SortsetDescription,
    SubjectDescription,
    to_serializable,
)
from kirby.utils import make_directory
from kirby.data import Interval, IrregularTimeSeries


class DatasetBuilder:
    def __init__(
        self,
        raw_folder_path: str,
        processed_folder_path: str,
        *,
        experiment_name: str,
        origin_version: str,
        derived_version: str,
        metadata_version: str = "0.0.2",
        source: str,
        description: str,
    ):
        self.raw_folder_path = raw_folder_path
        self.processed_folder_path = processed_folder_path

        self.experiment_name = experiment_name
        self.origin_version = origin_version
        self.derived_version = derived_version
        self.metadata_version = metadata_version
        self.source = source
        self.description = description

        # make processed folder if it doesn't exist
        # todo raise warning if it does exist, since some files might be overwritten
        make_directory(self.processed_folder_path, prompt_if_exists=False)

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

    def get_all_splits(self):
        """Return a list of all splits in the dataset"""
        splits = set()
        for sortset in self.sortsets:
            for session in sortset.sessions:
                for split in session.splits.keys():
                    splits.add(split)
        return list(splits)

    def finish(self):
        # Transform sortsets to a list of lists, otherwise it won't serialize to yaml.
        description = DandisetDescription(
            id=self.experiment_name,
            origin_version=self.origin_version,
            derived_version=self.derived_version,
            metadata_version=self.metadata_version,
            source=self.source,
            description=self.description,
            splits=self.get_all_splits(),
            subjects=self.subjects,
            sortsets=self.sortsets,
        )

        # Efficiently encode enums to strings
        description = to_serializable(description)

        # For efficiency, we also save a msgpack version of the description.
        # Smaller on disk, faster to read.
        filename = Path(self.processed_folder_path) / "description.mpk"
        print(f"Saving description to {filename}")

        with open(filename, "wb") as f:
            msgpack.dump(description, f, default=encode_datetime)


class SessionContextManager:
    def __init__(self, builder):
        self.builder = builder

        self.subject = None
        self.sortset = None
        self.session = None
        self.data = None

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
        units.unit_name = np.array([
            f"{self.builder.experiment_name}/{sortset_id}/{unit}"
            for unit in units.unit_name
        ])

        if sortset is None:
            sortset = SortsetDescription(
                id=id,
                subject=self.subject.id if self.subject is not None else "",
                sessions=sessions,
                units=units.unit_name.tolist(),
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

    def register_data(self, data):
        self.data = data
        self.register_split(
            "full",
            Interval(start=np.array([data.start]), end=np.array([data.end]))
        )

    def register_split(
        self,
        name: str,
        interval: Interval,
    ):
        """
        Args:
            name: name of the split
            interval: Interval object representing the split
        """
        if self.session.splits is None:
            self.session.splits = {}

        if name in self.session.splits:
            raise ValueError(f"Split {name} already exists for this session")

        # Can only handle Interval or list of tuples as split
        if not isinstance(interval, Interval):
            raise TypeError(f"Cannot handle interval type {type(interval)}")

        self.session.splits[name] = list(zip(interval.start, interval.end))
        self.data.add_split_mask(name, interval)

    def check_no_mask_overlap(self):
        """Performs a check on all split masks inside the data object to ensure
        there is no overlap across splits. Raises an error if there is overlap"""
        mask_names = [f"{x}_mask" for x in self.session.splits.keys() if x != "full"]
        for obj_key in self.data.keys:
            obj = getattr(self.data, obj_key)
            if isinstance(obj, (Interval, IrregularTimeSeries)):
                if isinstance(obj, Interval):
                    if obj._allow_split_mask_overlap:
                        continue

                mask_sum = np.zeros(len(obj))
                for mask_name in mask_names:
                    mask = getattr(obj, mask_name)
                    mask_sum += mask.astype(int)
                if np.any(mask_sum > 1):
                    if isinstance(obj, Interval):
                        raise ValueError(
                            f"Split mask overlap detected in {obj_key}. "
                            f"If you would like to allow overlap in this Interval "
                            f"object, call <object>.allow_split_mask_overlap() before "
                            f"saving session to disk."
                        )
                    else:
                        raise ValueError(
                            f"Split mask overlap detected in {obj_key}."
                        )

    def save_to_disk(self):
        assert self.subject is not None, "A subject must be registered."
        assert self.sortset is not None, "A sortset must be registered."
        assert self.session is not None, "A session must be registered."
        self.check_no_mask_overlap()

        path = os.path.join(self.builder.processed_folder_path, f"{self.session.id}.h5")

        with h5py.File(path, "w") as file:
            self.data.to_hdf5(file)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logging.error(f"Exception: {exc_type} {exc_value}")
            return False

        assert self.subject is not None, "A subject must be registered."
        assert self.sortset is not None, "A sortset must be registered."
        assert self.session is not None, "A session must be registered."

        # add subject to the dandiset if it hasn't been registered yet
        if not self.builder.is_subject_already_registered(self.subject.id):
            self.builder.subjects.append(self.subject)

        # add sortset to the dandiset if it hasn't been registered yet
        if not self.builder.is_sortset_already_registered(self.sortset.id):
            self.builder.sortsets.append(self.sortset)

        return True


def encode_datetime(obj):
    """msgpack doesn't support datetime, so we need to encode it as a string."""
    if isinstance(obj, datetime.datetime):
        return obj.strftime("%Y%m%dT%H:%M:%S.%f").encode()
