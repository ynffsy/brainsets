import collections
import datetime
from pathlib import Path
from typing import Any, Dict, List
import warnings

import msgpack
import yaml

from kirby.taxonomy.taxonomy import (
    ChunkDescription,
    DandisetDescription,
    SessionDescription,
    SortsetDescription,
    TrialDescription,
    to_serializable,
)


def encode_datetime(obj):
    """msgpack doesn't support datetime, so we need to encode it as a string."""
    if isinstance(obj, datetime.datetime):
        return obj.strftime("%Y%m%dT%H:%M:%S.%f").encode()


class DescriptionHelper:
    def __init__(self) -> None:
        self.dandiset = None
        self.sortsets = collections.defaultdict(list)
        self.sessions = collections.defaultdict(list)
        self.trials = collections.defaultdict(list)
        self.chunks = {}

    def register_dandiset(self, dandiset_description: DandisetDescription):
        self.dandiset = dandiset_description

    def register_sortset(
        self, dandiset_id: str, sortset_description: SortsetDescription
    ):
        self.sortsets[dandiset_id].append(sortset_description)

    def register_session(
        self, sortset_id: str, session_description: SessionDescription
    ):
        self.sessions[sortset_id].append(session_description)

    def register_trial(
        self, session_id: str, trial_description: TrialDescription
    ):
        self.trials[session_id].append(trial_description)

    def register_chunks(
        self, trial_id: str, chunks: Dict[str, List[ChunkDescription]]
    ):
        self.chunks[trial_id] = chunks

    def finalize(self) -> DandisetDescription:
        if self.dandiset is None:
            raise ValueError("Must register a dandiset first.")

        for key, value in self.chunks.items():
            self._attach_to_parent(self.trials, key, value, "chunks")

        for key, value in self.trials.items():
            self._attach_to_parent(self.sessions, key, value, "trials")

        for key, value in self.sessions.items():
            self._attach_to_parent(self.sortsets, key, value, "sessions")

        assert len(self.sortsets) == 1, "Only one dandiset is supported."
        self.dandiset.sortsets = list(self.sortsets.values())[0]
        return self.dandiset

    def _attach_to_parent(
        self,
        parent: Dict[str, Any],
        parent_id: str,
        value: Any,
        child_name: str,
    ):
        for el in parent.values():
            for el2 in el:
                if el2.id == parent_id:
                    # Set the value.
                    if getattr(el2, child_name):
                        raise NotImplementedError(
                            f"Multiple definitions of {child_name} over {parent_id}"
                        )
                    setattr(el2, child_name, value)

    def write_to_disk(self, path: Path, description: DandisetDescription):
        description = to_serializable(description)

        filename = Path(path) / "description.yaml"
        print(f"Saving description to {filename}")

        with open(filename, "w") as f:
            yaml.dump(description, f)

        # Check the description if it can be loaded.
        with open(filename, "r") as f:
            for i, line in enumerate(f.readlines()):
                if "!!" in line:
                    warnings.warn(
                        f"Found !! in description: at line {i+1}\n\t {line}"
                    )

        # For efficiency, we also save a msgpack version of the description.
        # Smaller on disk, faster to read.
        filename = Path(path) / "description.mpk"
        print(f"Saving description to {filename}")

        with open(filename, "wb") as f:
            msgpack.dump(description, f, default=encode_datetime)
