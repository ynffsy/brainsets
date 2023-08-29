import dataclasses
import datetime
import collections
from enum import Enum
from typing import List, Dict
import numpy as np

from dataclasses import dataclass, asdict


class StringIntEnum(Enum):
    """Enum where the value is a string, but can be cast to an int."""

    def __str__(self):
        return self.name

    def __int__(self):
        return self.value


class RecordingTech(StringIntEnum):
    UTAH_ARRAY_SPIKES = 0
    UTAH_ARRAY_THRESHOLD_CROSSINGS = 1
    UTAH_ARRAY_WAVEFORMS = 2
    UTAH_ARRAY_LFPS = 3

    # As a subordinate category
    UTAH_ARRAY = 9


class Task(StringIntEnum):
    # A classic BCI task involving reaching to a 2d target.
    DISCRETE_REACHING = 0

    # A continuous version of the classic BCI without discrete trials.
    CONTINUOUS_REACHING = 1


class Stimulus(StringIntEnum):
    """Stimuli can variously act like inputs (for conditioning) or like outputs."""

    TARGET2D = 0


class Output(StringIntEnum):
    # Classic BCI outputs.
    ARM2D = 0
    CURSOR2D = 1
    EYE2D = 2
    FINGER3D = 3

    DISCRETE_TRIAL_ONSET_OFFSET = 10
    CONTINUOUS_TRIAL_ONSET_OFFSET = 11


class Species(StringIntEnum):
    MACACA_MULATTA = 0


class Dictable:
    """A dataclass that can be converted to a dict."""

    def to_dict(self):
        """__dict__ doesn't play well with torch.load"""
        return {k: v for k, v in asdict(self).items()}  # type: ignore


@dataclass
class ChunkDescription(Dictable):
    id: str
    duration: float
    start_time: float  # Relative to start of trial.


@dataclass
class TrialDescription(Dictable):
    id: str
    footprints: Dict[str, int]
    chunks: Dict[str, List[ChunkDescription]]


@dataclass
class SessionDescription(Dictable):
    id: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    task: Task
    inputs: Dict[RecordingTech, str]
    stimuli: Dict[Stimulus, str]
    outputs: Dict[Output, str]
    trials: List[TrialDescription]


@dataclass
class SortsetDescription(Dictable):
    id: str
    subject: str
    areas: List[StringIntEnum]
    recording_tech: List[RecordingTech]
    sessions: List[SessionDescription]
    units: List[str]

@dataclass 
class SubjectDescription(Dictable):
    id: str
    species: Species
    age: float = 0.0

@dataclass
class DandisetDescription(Dictable):
    id: str
    origin_version: str
    derived_version: str
    metadata_version: str
    source: str
    description: str
    folds: List[str]
    subjects: List[SubjectDescription]
    sortsets: List[SortsetDescription]


def to_serializable(dct):
    """Recursively map data structure elements to string when they are of type
    StringIntEnum"""
    if isinstance(dct, list):
        return [to_serializable(x) for x in dct]
    elif isinstance(dct, dict) or isinstance(dct, collections.defaultdict):
        return {to_serializable(x): to_serializable(y) for x, y in dict(dct).items()}
    elif isinstance(dct, Dictable):
        return {
            x.name: to_serializable(getattr(dct, x.name))
            for x in dataclasses.fields(dct)
        }
    elif isinstance(dct, StringIntEnum):
        return str(dct)
    elif isinstance(dct, np.ndarray):
        if np.isscalar(dct):
            return dct.item()
        else:
            raise NotImplementedError("Cannot serialize numpy arrays.")
    elif (
        isinstance(dct, str)
        or isinstance(dct, int)
        or isinstance(dct, float)
        or isinstance(dct, bool)
        or isinstance(dct, type(None))
        or isinstance(dct, datetime.datetime)
    ):
        return dct
    else:
        raise NotImplementedError(f"Cannot serialize {type(dct)}")
