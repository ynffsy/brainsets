import collections
import dataclasses
import datetime
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

from pydantic.dataclasses import dataclass
import torch

from kirby.taxonomy.macaque import Macaque

from .core import StringIntEnum
from .writing import Character, Line
from .speech import CVSyllable

class RecordingTech(StringIntEnum):
    UTAH_ARRAY_SPIKES = 0
    UTAH_ARRAY_THRESHOLD_CROSSINGS = 1
    UTAH_ARRAY_WAVEFORMS = 2
    UTAH_ARRAY_LFPS = 3
    UTAH_ARRAY_AVERAGE_WAVEFORMS = 4
    # As a subordinate category
    UTAH_ARRAY = 9

    NEUROPIXELS_SPIKES = 10
    NEUROPIXELS_THRESHOLDCROSSINGS = 11
    NEUROPIXELS_WAVEFORMS = 12
    NEUROPIXELS_LFPS = 13
    # As a subordinate category
    NEUROPIXELS_ARRAY = 19


    ECOG_ARRAY_ECOGS = 29


class Task(StringIntEnum):
    # A classic BCI task involving reaching to a 2d target.
    DISCRETE_REACHING = 0

    # A continuous version of the classic BCI without discrete trials.
    CONTINUOUS_REACHING = 1

    # A Shenoy-style task involving handwriting different characters.
    DISCRETE_WRITING_CHARACTER = 2

    DISCRETE_WRITING_LINE = 3

    CONTINUOUS_WRITING = 4

    # Allen data
    DISCRETE_VISUAL_CODING = 5

    # speech
    DISCRETE_SPEAKING_CVSYLLABLE = 6


class Stimulus(StringIntEnum):
    """Stimuli can variously act like inputs (for conditioning) or like outputs."""

    TARGET2D = 0
    TARGETON = 1
    GO_CUE = 2
    TARGETACQ = 3

    DRIFTING_GRATINGS = 4


class Output(StringIntEnum):
    # Classic BCI outputs.
    ARMVELOCITY2D = 0
    CURSOR2D = 1
    EYE2D = 2
    FINGER3D = 3

    # Shenoy handwriting style outputs.
    WRITING_CHARACTER = 4
    WRITING_LINE = 5

    DISCRETE_TRIAL_ONSET_OFFSET = 10
    CONTINUOUS_TRIAL_ONSET_OFFSET = 11

    CURSORVELOCITY2D = 12

    # Allen data
    DRIFTING_GRATINGS = 13

    # speech
    SPEAKING_CVSYLLABLE = 14


class Species(StringIntEnum):
    MACACA_MULATTA = 0
    HOMO_SAPIENS = HUMAN = 1
    MUS_MUSCULUS = 2


class Sex(StringIntEnum):
    r"""Follows the DANDI definition of sex.
    [Link](https://www.dandiarchive.org/handbook/135_validation/#missing-dandi-metadata)
    """
    UNKNOWN = U = 0
    MALE = M = 1
    FEMALE = F = 2
    OTHER = O = 3


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
    recording_date: datetime.datetime
    task: Task
    fields: Dict[Union[RecordingTech, Stimulus, Output], str]
    trials: List[TrialDescription]
    splits: Optional[Dict[str, List[Tuple[float, float]]]] = None
    start_time: Optional[datetime.datetime] = None  # todo: deprecate
    end_time: Optional[datetime.datetime] = None  # todo: deprecate


@dataclass
class SortsetDescription(Dictable):
    id: str
    subject: str
    areas: Union[List[StringIntEnum], List[Macaque]]
    recording_tech: List[RecordingTech]
    sessions: List[SessionDescription]
    units: List[str]


@dataclass
class SubjectDescription(Dictable):
    id: str
    species: Species
    age: float = 0.0  # in days
    sex: Sex = Sex.UNKNOWN
    genotype: str = "unknown"  # no idea how many there will be for now.


@dataclass
class DandisetDescription(Dictable):
    id: str
    origin_version: str
    derived_version: str
    metadata_version: str
    source: str
    description: str
    splits: List[str]
    subjects: List[SubjectDescription]
    sortsets: List[SortsetDescription]


def to_serializable(dct):
    """Recursively map data structure elements to string when they are of type
    StringIntEnum"""
    if isinstance(dct, list) or isinstance(dct, tuple):
        return [to_serializable(x) for x in dct]
    elif isinstance(dct, dict) or isinstance(dct, collections.defaultdict):
        return {
            to_serializable(x): to_serializable(y)
            for x, y in dict(dct).items()
        }
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

class OutputType(StringIntEnum):
    CONTINUOUS = 0
    BINARY = 1
    MULTILABEL = 2
    MULTINOMIAL = 3

@dataclass
class DecoderSpec:
    dim: int
    target_dim: int  # For multilabel and multinomial, target_dim can be 1 when the output is an int.
    # For soft classes instead, target_dim = dim.
    type: OutputType
    timestamp_key: str
    value_key: str
    loss_fn: str
    tag_key: Optional[str] = None
    target_dtype: str = "float32"  # torch.dtype is not serializable.
    behavior_type_key = "behavior.type"


decoder_registry = {
    str(Output.ARMVELOCITY2D) : DecoderSpec(dim=2, 
                                            target_dim=2,
                                       type=OutputType.CONTINUOUS, 
                                       timestamp_key="behavior.timestamps",
                                       value_key="behavior.hand_vel", 
                                       loss_fn="mse",
                                      ),
    str(Output.CURSORVELOCITY2D) : DecoderSpec(dim=2, 
                                               target_dim=2,
                                       type=OutputType.CONTINUOUS, 
                                       timestamp_key="behavior.timestamps",
                                       value_key="behavior.cursor_vel", 
                                       loss_fn="mse",
                                      ),
    str(Output.CURSOR2D) : DecoderSpec(dim=2, 
                                       target_dim=2,
                                       type=OutputType.CONTINUOUS, 
                                       timestamp_key="behavior.timestamps",
                                       value_key="behavior.cursor_pos", 
                                       loss_fn="mse",
                                      ),
    str(Output.WRITING_CHARACTER) : DecoderSpec(dim=len(Character), 
                                     target_dim=1,
                                     target_dtype="long",
                                     type=OutputType.MULTINOMIAL, 
                                     timestamp_key="stimuli_segments.timestamps",
                                     value_key="stimuli_segments.letters", 
                                     loss_fn="bce",
                                    ),
    str(Output.WRITING_LINE) : DecoderSpec(
                                    dim=len(Line), 
                                    target_dim=1,
                                    target_dtype="long",
                                    type=OutputType.MULTINOMIAL, 
                                    timestamp_key="stimuli_segments.timestamps",
                                    value_key="stimuli_segments.letters", 
                                    loss_fn="bce",
                                ),
    str(Output.DRIFTING_GRATINGS) : DecoderSpec(dim=8, 
                                    target_dim=1,
                                    target_dtype="long",
                                    type=OutputType.MULTINOMIAL, 
                                    timestamp_key="stimuli_segments.timestamps",
                                    value_key="stimuli_segments.drifting_class", 
                                    loss_fn="bce",
                                ),
    str(Output.SPEAKING_CVSYLLABLE) : DecoderSpec(dim=len(CVSyllable), # empty label is included 
                                    target_dim=1,
                                    target_dtype="long",
                                    type=OutputType.MULTINOMIAL, 
                                    timestamp_key="speech.timestamps",
                                    value_key="speech.consonant_vowel_syllables", 
                                    loss_fn="bce",
                                ),
}
