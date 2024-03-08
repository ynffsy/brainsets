from typing import Dict, List, Tuple, Optional, Union, Any

from pydantic.dataclasses import dataclass

from .core import StringIntEnum


class OutputType(StringIntEnum):
    CONTINUOUS = 0
    BINARY = 1
    MULTILABEL = 2
    MULTINOMIAL = 3


class Decoder(StringIntEnum):
    NA = 0
    # Classic BCI outputs.
    ARMVELOCITY2D = 1
    CURSOR2D = 2
    EYE2D = 3
    FINGER3D = 4

    # Shenoy handwriting style outputs.
    WRITING_CHARACTER = 5
    WRITING_LINE = 6

    DISCRETE_TRIAL_ONSET_OFFSET = 7
    CONTINUOUS_TRIAL_ONSET_OFFSET = 8

    CURSORVELOCITY2D = 9

    # Allen data
    DRIFTING_GRATINGS = 13
    DRIFTING_GRATINGS_TEMP_FREQ = 23

    # Openscope calcium
    UNEXPECTED_OR_NOT = 20  #
    GABOR_ORIENTATION = 21  #
    PUPIL_MOVEMENT_REGRESSION = 22

    # speech
    SPEAKING_CVSYLLABLE = 14


@dataclass
class DecoderSpec:
    dim: int
    type: OutputType
    loss_fn: str
    timestamp_key: str
    value_key: str
    # Optional fields
    task_key: Optional[str] = None
    subtask_key: Optional[str] = None
    # target_dtype: str = "float32"  # torch.dtype is not serializable.


decoder_registry = {
    str(Decoder.ARMVELOCITY2D): DecoderSpec(
        dim=2,
        target_dim=2,
        type=OutputType.CONTINUOUS,
        timestamp_key="behavior.timestamps",
        value_key="behavior.hand_vel",
        subtask_key="behavior.subtask_index",
        loss_fn="mse",
    ),
    str(Decoder.CURSORVELOCITY2D): DecoderSpec(
        dim=2,
        target_dim=2,
        type=OutputType.CONTINUOUS,
        timestamp_key="cursor.timestamps",
        value_key="cursor.vel",
        subtask_key="cursor.subtask_index",
        loss_fn="mse",
    ),
    str(Decoder.CURSOR2D): DecoderSpec(
        dim=2,
        target_dim=2,
        type=OutputType.CONTINUOUS,
        timestamp_key="behavior.timestamps",
        value_key="behavior.cursor_pos",
        loss_fn="mse",
    ),
    # str(Decoder.WRITING_CHARACTER): DecoderSpec(
    #     dim=len(Character),
    #     target_dim=1,
    #     target_dtype="long",
    #     type=OutputType.MULTINOMIAL,
    #     timestamp_key="stimuli_segments.timestamps",
    #     value_key="stimuli_segments.letters",
    #     loss_fn="bce",
    # ),
    # str(Decoder.WRITING_LINE): DecoderSpec(
    #     dim=len(Line),
    #     target_dim=1,
    #     target_dtype="long",
    #     type=OutputType.MULTINOMIAL,
    #     timestamp_key="stimuli_segments.timestamps",
    #     value_key="stimuli_segments.letters",
    #     loss_fn="bce",
    # ),
    str(Decoder.DRIFTING_GRATINGS): DecoderSpec(
        dim=8,
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="stimuli_segments.timestamps",
        value_key="stimuli_segments.drifting_class",
        loss_fn="bce",
    ),
    str(Decoder.DRIFTING_GRATINGS_TEMP_FREQ): DecoderSpec(
        dim=5,  # [1,2,4,8,15]
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="stimuli_segments.timestamps",
        value_key="stimuli_segments.drifting_temp_freq",
        loss_fn="bce",
    ),
    # str(Decoder.SPEAKING_CVSYLLABLE): DecoderSpec(
    #     dim=len(CVSyllable),  # empty label is included
    #     target_dim=1,
    #     target_dtype="long",
    #     type=OutputType.MULTINOMIAL,
    #     timestamp_key="speech.timestamps",
    #     value_key="speech.consonant_vowel_syllables",
    #     loss_fn="bce",
    # ),
    str(Decoder.GABOR_ORIENTATION): DecoderSpec(
        dim=4,  # [0, 1, 2, 3]
        target_dim=1,
        target_dtype="long",
        type=OutputType.MULTINOMIAL,
        timestamp_key="gabor_trials.timestamps",
        value_key="gabor_trials.gabor_orientation",
        loss_fn="bce",
    ),
}
