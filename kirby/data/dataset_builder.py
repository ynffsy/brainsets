import datetime
import logging
import msgpack
import os
from pathlib import Path
import yaml
from typing import (
    List,
    Dict,
    Optional,
    Union,
    Tuple,
)

import h5py
import numpy as np
from kirby.taxonomy.core import StringIntEnum

from kirby.taxonomy import (
    DandisetDescription,
    SessionDescription,
    SortsetDescription,
    SubjectDescription,
    TrialDescription,
    to_serializable,
    Task, RecordingTech, Stimulus, Output, Sex,
    Macaque,
)
from kirby.utils import make_directory
from kirby.data import Interval, IrregularTimeSeries, ArrayDict, Data


class DatasetBuilder:
    r"""A class to help build a standardized dataset.
     
    Args:
        raw_folder_path: The path to the raw data folder.
        processed_folder_path: The path to the processed data folder.
        experiment_name: The name of the experiment.
        origin_version: The version of the data depending on source (dandi version, 
            zenodo version, etc). If version is unknown, use "unknown".
        derived_version: The version of the data after processing, this is a unique
            identifier for the processed data, incase you need to have multiple versions
            of the processed data.
        metadata_version: The version of the metadata, this should be the version of 
            our package, this will be deprecated and set automatically. Defaults to 
            "0.0.2".
        source: The source of the data. This is the link to the data source (url), if 
            the data is not public, add a description of the data source.
        description: A description of the data.

    .. code-block:: python
    
            from kirby.data import DatasetBuilder
    
            builder = DatasetBuilder(
                raw_folder_path="/path/to/raw",
                processed_folder_path="/path/to/processed",
                experiment_name="my_experiment",
                origin_version="unknown",
                derived_version="0.0.1",
                source="https://example.com",
                description="This is a description of the data."
            )
    """
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
        # check that the raw folder exists
        if not os.path.exists(raw_folder_path):
            raise ValueError(f"Folder {raw_folder_path} does not exist.")

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

        self.subjects: List[SubjectDescription] = []
        self.sortsets: List[SortsetDescription] = []

    def new_session(self):
        r"""Start a new :obj:`SessionContextManager`, which will help collect the data 
        and metadata for a new session.
        
        .. warning::
            This method should be used as a context manager, so it should be used with 
            the `with` statement. This is to ensure that the session is properly 
            registered to the dandiset after data collection is complete.

            .. code-block:: python
                
                with builder.new_session() as session:
                    ...
        """
        # initialize the session
        # each session should have 1 subject and 1 sortset
        return SessionContextManager(self)

    def is_subject_already_registered(self, subject_id):
        r"""Check if a subject is already registered to the dandiset."""
        # register subject to the dandiset if it hasn't been registered yet
        return any([subject_id == member.id for member in self.subjects])

    def is_sortset_already_registered(self, sortset_id):
        r"""Check if a sortset is already registered to the dandiset."""
        # Check if the sortset is already registered
        return any([sortset_id == member.id for member in self.sortsets])

    def get_sortset(self, sortset_id):
        r"""Get the (:obj:`SortestDescription`) of a sortset by its id."""
        return next(
            (member for member in self.sortsets if member.id == sortset_id), None
        )

    def get_subject(self, subject_id):
        r"""Get the (:obj:`SubjectDescription`) of a subject by its id."""
        return next(
            (member for member in self.subjects if member.id == subject_id), None
        )

    def get_all_sessions(self):
        """Return a list of all sessions in the dataset"""
        return sum([sortset.sessions for sortset in self.sortsets], [])

    def get_all_splits(self):
        """Return a list of all splits in the dataset"""
        splits = set()
        for sortset in self.sortsets:
            for session in sortset.sessions:
                for split in session.splits.keys():
                    splits.add(split)
        return list(splits)

    def finish(self):
        r"""Save the dandiset description to disk. This should be called after all
        sessions have been registered.
        
        .. code-block:: python
            
            builder.finish()
        """
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
    r"""A context manager to help collect the data and metadata for a new session."""
    def __init__(self, builder):
        self.builder: DatasetBuilder = builder

        self.subject: Optional[SubjectDescription] = None
        self.sortset: Optional[SortsetDescription] = None
        self.session: Optional[SessionDescription] = None
        self.data: Optional[Data] = None

    def __enter__(self):
        return self

    def register_subject(
        self, 
        subject: SubjectDescription = None,
        *,
        id: str = None,
        species: str = None,
        age: float = 0.0,
        sex: Sex = Sex.UNKNOWN,
        genotype: str = "unknown",
    ):
        """Register subject metadata onto the session context manager.

        Args:
            subject: A :class:`~kirby.taxonomy.SubjectDescription` object containing
                the subject metadata. Either provide this, or the following arguments.
            id: A sortset identifier string. Must be unique within the dandiset.
                Must be provided if `sortset` argument is not provided.
            species: A string representing the species of the subject.
                Must be provided if `sortset` argument is not provided.
            age (optional): The age of the subject in days.
            sex (optional): A :class:`~kirby.taxonomy.Sex` enum.
            genotype (optional): A string representing the genotype of the subject.

        .. code-block:: python

            subject = extract_subject_from_nwb(nwbfile)
            session.register_subject(subject)

        """

        if self.subject is not None:
            raise ValueError(
                "A subject was already registered. A session can only have "
                "one subject."
            )

        if subject is None:
            assert id is not None, "Subject id must be provided"
            assert species is not None, "Subject species must be provided"
            subject = SubjectDescription(
                id=id,
                species=species,
                age=age,
                sex=sex,
                genotype=genotype,
            )

        self.subject = subject

        # if sortset was defined before subject, we need to update the reference
        if self.sortset is not None:
            self.sortset.subject = self.subject.id

    def register_sortset(
        self,
        sortset: SortsetDescription = None,
        *,
        id: str = None,
        units: List[str],
        areas: Union[List[StringIntEnum], List[Macaque]] = [],
        recording_tech: List[RecordingTech] = [],
    ):
        """Register sortset metadata onto the session context manager.

        Args:
            sortset: A :class:`~kirby.taxonomy.SortsetDescription` object.
                Either provide this, or the following arguments.
            id: A sortset identifier string. Must be unique within the dandiset.
                Must be provided if `sortset` argument is not provided.
            units: A list of unit identifiers. These unit-ids must be unique within
                the dandiset. Must always be provided.
            areas (optional): A list of :class:`~kirby.taxonomy.StringIntEnum` or
                :class:`~kirby.taxonomy.Macaque` enums.
            recording_tech (optional): A list of :class:`~kirby.taxonomy.RecordingTech` 
                enums.

        .. code-block:: python

            spikes, units = extract_spikes_from_nwbfile(
                nwbfile,
                recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
            )

            session_context_manager.register_sortset(
                id="jenkins_20090928",
                units=units,
            )

        """

        if self.sortset is not None:
            raise ValueError(
                "A sortset was already registered. A session can only have "
                "one sortset."
            )

        # add prefix to unit names
        units.id = np.array(
            [f"{self.builder.experiment_name}/{id}/{unit_id}" for unit_id in units.id]
        )

        if sortset is None:
            assert id is not None, "Sortset id must be provided"
            assert units is not None, "Sortset units must be provided"
            sortset = SortsetDescription(
                id=id,
                subject=self.subject.id if self.subject else "",
                sessions=[], # will be filled by register_session(...)
                units=units.id.tolist(),
                areas=areas,
                recording_tech=recording_tech,
            )
        else:
            sortset.units = units.id.tolist()
            sortset.sessions = []

        # Check if the sortset is already registered
        existing_sortset = self.builder.get_sortset(sortset.id)

        if existing_sortset is None:
            self.sortset = sortset
        else:
            # If it exists, make sure all the properties match and
            # update the reference
            for key in sortset.as_dict().keys():
                if (key != "sessions" and  # sessions list is not expected to match
                    (getattr(existing_sortset, key) != getattr(sortset, key))
                ):
                    raise ValueError(
                        f"Sortset {sortset.id} has already been registered "
                        f"with different properties. Mismatch at key {key}. "
                        f"Existing: {getattr(existing_sortset, key)}, "
                        f"New: {getattr(sortset, key)}"
                    )
            self.sortset = existing_sortset

        # if session was defined before sortset, we need to update the reference
        if self.session is not None:
            self.sortset.sessions.append(self.session)

    def register_session(
        self, 
        session: SessionDescription = None,
        *,
        id: str = None,
        recording_date: datetime.datetime = None,
        task: Task = None,
        fields: Dict[Union[RecordingTech, Stimulus, Output], str] = None,
        trials: List[TrialDescription] = []
    ) -> None:
        """Register session metadata onto the context manager.

        Args:
            session: A :class:`~kirby.taxonomy.SessionDescription` object.
                Either provide this, or the following arguments.
            id: A session identifier string. Must be unique within the dandiset.
                Must be provided if `session` argument is not provided.
            recording_date: A datetime object representing the date of the recording.
                Must be provided if `session` argument is not provided.
            task: A :class:`~kirby.taxonomy.Task` enum
                Must be provided if `session` argument is not provided.
            fields: A dictionary mapping 
                :class:`~kirby.taxonomy.RecordingTech`, 
                :class:`~kirby.taxonomy.Stimulus`,
                or :class:`~kirby.taxonomy.Output` enums to strings marking the
                field names in the data object corresponding to that type of data.
                Must be provided if `session` argument is not provided.
            trials (optional): A list of :class:`~kirby.taxonomy.TrialDescription`
                objects.

        .. code-block:: python

            session_context_manager.register_session(
                id="jenkins_20090928_maze",
                recording_date=datetime.datetime.strptime("20090928", "%Y%m%d"),
                task=Task.DISCRETE_REACHING,
                fields={
                    RecordingTech.UTAH_ARRAY_SPIKES: "spikes",
                    Output.CURSOR2D: "behavior.hand_vel",
                },
            )

        """
        if self.session is not None:
            raise ValueError(
                "A session was already registered. "
                "You can only register one session per session context."
            )

        if session is None:
            assert id is not None, "Session id must be provided"
            assert recording_date is not None, "Session recording date must be provided"
            assert task is not None, "Session task must be provided"
            assert fields is not None, "Session fields must be provided"
            session = SessionDescription(
                id=id,
                recording_date=recording_date,
                task=task,
                fields=fields,
                trials=trials,
                splits={}, # Will fill in register_split(...)
                dandiset_id=None, # Will fill in __exit__
                subject_id=None, # Will fill in __exit__
                sortset_id=None, # Will fill in __exit__
            )
        else:
            session.splits = {}
            session.dandiset_id = None
            session.subject_id = None
            session.sortset_id = None

        # Ensure id is unique within entire dandiset
        for existing_session in self.builder.get_all_sessions():
            if existing_session.id == session.id:
                raise ValueError(
                    f"Session with id {session.id} already exists. "
                    f"Session ids must be unique within the entire dandiset."
                )

        
        self.session = session

        # If sortset was defined before session, we need to update the reference
        if self.sortset is not None:
            self.sortset.sessions.append(self.session)

    def register_data(
        self, 
        data: Data
    ):
        """Register a :class:`~kirby.data.Data` object for this session."""

        assert self.data is None, (
            "A data object was already registered. "
            "You can only register one data object per session."
        )

        self.data = data
        self.register_split("full", self.data.domain)

    def register_split(
        self,
        name: str,
        interval: Interval,
    ):
        """
        Args:
            name: name of the split, eg. standard names:
                "train", "test", "valid" (for validation)
            interval: :class:`kirby.data.Interval` object defining the split
        """
        if self.session is None:
            raise ValueError("A session must be registered before registering splits")

        if self.session.splits is None:
            self.session.splits = {}

        if name in self.session.splits:
            raise ValueError(f"Split {name} already exists for this session")

        # Can only handle Interval or list of tuples as split
        if not isinstance(interval, Interval):
            raise TypeError(f"Cannot handle type {type(interval)}")

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
        assert self.data is not None, "A data object must be registered."

        self.data.subject_id = self.subject.id
        self.data.session_id = self.session.id
        self.data.sortset_id = self.sortset.id

        self.session.dandiset_id = self.builder.experiment_name
        self.session.subject_id = self.subject.id
        self.session.sortset_id = self.sortset.id

        # add subject to the dandiset if it hasn't been registered yet.
        # if this subject has already been registered, we need to make sure all the
        # properties match
        if not self.builder.is_subject_already_registered(self.subject.id):
            self.builder.subjects.append(self.subject)
        else:
            existing_subject = self.builder.get_subject(self.subject.id)
            # todo: there is a bug here, we need to check if the subject is already
            # for key in self.subject.as_dict().keys():
            #     if (getattr(existing_subject, key) != getattr(self.subject, key)):
            #         raise ValueError(
            #             f"Subject {self.subject.id} has already been registered "
            #             f"with different properties. Mismatch at key {key}. "
            #             f"Existing: {getattr(existing_subject, key)}, "
            #             f"New: {getattr(self.subject, key)}"
            #             )

        # add sortset to the dandiset if it hasn't been registered yet
        if not self.builder.is_sortset_already_registered(self.sortset.id):
            self.builder.sortsets.append(self.sortset)

        return True


def encode_datetime(obj):
    """msgpack doesn't support datetime, so we need to encode it as a string."""
    if isinstance(obj, datetime.datetime):
        return obj.strftime("%Y%m%dT%H:%M:%S.%f").encode()
