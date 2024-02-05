import numpy as np

from kirby.data import (
    Data,
    IrregularTimeSeries,
)

from kirby.taxonomy import (
    RecordingTech,
    Sex,
    Species,
    SubjectDescription,
)


def extract_metadata_from_nwb(nwbfile):
    recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
    related_publications = nwbfile.related_publications
    return dict(
        recording_date=recording_date, related_publications=related_publications
    )


def extract_subject_from_nwb(nwbfile):
    r"""DANDI has requirements for metadata included in `subject`. This includes:
    subject_id: A subject identifier must be provided.
    species: either a latin binomial or NCBI taxonomic identifier.
    sex: must be "M", "F", "O" (other), or "U" (unknown).
    date_of_birth or age: this does not appear to be enforced, so will be skipped.
    """
    return SubjectDescription(
        id=nwbfile.subject.subject_id.lower(),
        species=Species.from_string(nwbfile.subject.species),
        sex=Sex.from_string(nwbfile.subject.sex),
    )


def extract_spikes_from_nwbfile(nwbfile, recording_tech):
    # spikes
    timestamps = []
    unit_index = []

    # units
    unit_meta = []

    units = nwbfile.units.spike_times_index[:]
    electrodes = nwbfile.units.electrodes.table

    # all these units are obtained using threshold crossings
    for i in range(len(units)):
        # label unit
        group_name = electrodes["group_name"][i]
        unit_name = f"group_{group_name}/unit_{i}"
        
        # extract spikes
        spiketimes = units[i]
        timestamps.append(spiketimes)
        unit_index.append([i] * len(spiketimes))

        # extract unit metadata
        unit_meta.append(
            {
                "unit_name": unit_name,
                "unit_number": i,
                "count": len(spiketimes),
                "type": int(recording_tech),
            }
        )


    # convert unit metadata to a Data object
    # Cast to np.ndarray
    unit_meta_long = {}
    for key, item in unit_meta[0].items():
        stacked_array = np.stack([x[key] for x in unit_meta], axis=0)
        if np.issubdtype(type(item), np.number):
            if np.issubdtype(type(item), np.unsignedinteger):
                stacked_array = stacked_array.astype(np.int64)
            unit_meta_long[key] = np.array(stacked_array)
        else:
            unit_meta_long[key] = stacked_array
    units = Data(**unit_meta_long)

    # concatenate spikes
    timestamps = np.concatenate(timestamps)
    unit_index = np.concatenate(unit_index)

    # create spikes object
    spikes = IrregularTimeSeries(
        timestamps=timestamps,
        unit_index=unit_index,
    )

    # make sure to sort ethe spikes
    spikes.sort()

    return spikes, units
