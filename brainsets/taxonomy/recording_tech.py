from dataclasses import dataclass

from ..core import StringIntEnum, Dictable


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

    OPENSCOPE_CALCIUM_TRACES = 20
    OPENSCOPE_CALCIUM_RAW = 21

    ECOG_ARRAY_ECOGS = 29
    MICRO_ECOG_ARRAY_ECOGS = 30

    TWO_PHOTON_IMAGING = 40


class Hemisphere(StringIntEnum):
    UNKNOWN = 0
    LEFT = 1
    RIGHT = 2


class ImplantArea:
    array_int_to_str = {
        0: "MC",
        1: "MC-LAT",
        2: "MC-MED",
        11: "PPC",
        12: "PPC-SPL",
        13: "PPC-IPL",
        }

    array_str_to_int = {
        "MC": 0,
        "MC-LAT": 1,
        "MC-MED": 2,
        "PPC": 11,
        "PPC-SPL": 12,
        "PPC-IPL": 13,
    }


# @dataclass
# class Channel(Dictable):
#     """Channels are the physical channels used to record the data. Channels are grouped
#     into probes."""

#     id: str
#     local_index: int

#     # Position relative to the reference location of the probe, in microns.
#     relative_x_um: float
#     relative_y_um: float
#     relative_z_um: float

#     area: StringIntEnum
#     hemisphere: Hemisphere = Hemisphere.UNKNOWN


# @dataclass
# class Probe(Dictable):
#     """Probes are the physical probes used to record the data."""

#     id: str
#     type: RecordingTech
#     lfp_sampling_rate: float
#     wideband_sampling_rate: float
#     waveform_sampling_rate: float
#     waveform_samples: int
#     channels: list[Channel]
#     ecog_sampling_rate: float = 0.0
