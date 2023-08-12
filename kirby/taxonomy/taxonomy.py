import datetime
from enum import Enum, IntEnum

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

class Task(StringIntEnum):
    # A classic BCI task involving reaching to a 2d target.
    REACHING_TRIAL = 0

    # A continuous version of the classic BCI without discrete trials.
    REACHING_CONTINUOUS = 1

class Output(StringIntEnum):
    # Classic BCI outputs.
    ARM2D = 0
    CURSOR2D = 1
    EYE2D = 2
    FINGER3D = 3
    TARGET2D = 4

    DISCRETE_TRIAL_ONSET_OFFSET = 10
    CONTINUOUS_TRIAL_ONSET_OFFSET = 11

class Dictable():
    """A dataclass that can be converted to a dict."""
    def __dict__(self):
        return {k: v for k, v in asdict(self).items()} # type: ignore

@dataclass
class Session(Dictable):
    subject: str
    date: datetime.date
    configuration: str
    train_footprint: list[dict[str, int]]
    valid_footprint: list[dict[str, int]]
    test_footprint: list[dict[str, int]]