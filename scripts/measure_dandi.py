import json
from pathlib import Path

import fsspec
import h5py
import numpy as np
import pynwb
import requests
import tqdm
from dandi.dandiapi import DandiAPIClient
from fsspec.implementations.cached import CachingFileSystem
from joblib import Parallel, delayed


def custom_serializer(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, np.uint64):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def find_spike_info(nwbfile, has_lfp_or_ca):
    recording_length = 0
    if nwbfile.units is None or nwbfile.units.spike_times is None:
        nunits = 0
        nspikes = 0
    else:
        attributes = ["id", "global_id", "origClusterID", "electrodes"]
        nspikes = len(nwbfile.units.spike_times)

        nunits = 1
        # Try one of these N attributes:
        for attr in attributes:
            if getattr(nwbfile.units, attr) is not None:
                nunits = len(nwbfile.units[attr])
                break

        if nunits == 0:
            raise Exception("No units found")

        if not has_lfp_or_ca:
            # This requires downloading the file, so only do it if we don't have LFP or
            # calcium.
            recording_length = (
                nwbfile.units.spike_times[-1] - nwbfile.units.spike_times[0]
            )

    if nunits == 0:
        # Last ditch effort: look for evidence of binned spikes.
        # Look for a time series called rasters or firing rates.
        keywords = ["raster", "firing_rate", "spike_count"]

        def look_for_spikes(acquisition, prefix=""):
            for k, v in acquisition.items():
                if isinstance(v, pynwb.base.TimeSeries) and any(
                    [x in k.lower() for x in keywords]
                ):
                    return v
                elif isinstance(v, pynwb.base.ProcessingModule):
                    result = look_for_spikes(v.data_interfaces, prefix=prefix + k + ".")
                    if result is not None:
                        return result
            return None

        raster = look_for_spikes(nwbfile.processing)
        if raster is None:
            raster = look_for_spikes(nwbfile.acquisition)

        if raster is not None:
            # These are not sparse spikes, but rather dense binned spikes.
            rate = raster.rate
            if rate < 0.1:
                rate = 1 / rate

            # We would need to add up the number of spikes in each raster
            recording_length = raster.data.shape[0] / rate
            nunits = raster.data.shape[1]
            nspikes = np.array(raster.data).sum()

    info = {
        "nunits": nunits,
        "nspikes": nspikes,
        "spike_recording_length": recording_length,
    }

    return info


def find_lfp_info(nwbfile):
    """Look for LFP data in the NWB file.

    Look for a pynwb.ecephys.LFP or pynwb.ecephys.ElectricalSeries object
    """

    def look_for_lfp(acquisition, prefix=""):
        for k, v in acquisition.items():
            if isinstance(v, pynwb.ecephys.LFP):
                for k2, v2 in v.electrical_series.items():
                    if isinstance(v2, pynwb.ecephys.ElectricalSeries):
                        return v2
            if isinstance(v, pynwb.ecephys.ElectricalSeries):
                return v
            if isinstance(v, pynwb.base.TimeSeries) and "lfp" in k.lower():
                # Naked time series with the right name
                return v
            elif isinstance(v, pynwb.base.ProcessingModule):
                result = look_for_lfp(v.data_interfaces, prefix=prefix + k + ".")
                if result is not None:
                    return result
        return None

    lfp = look_for_lfp(nwbfile.processing, "processing.")

    if lfp is None:
        lfp = look_for_lfp(nwbfile.acquisition, "acquisition.")

    if lfp is not None:
        shape = lfp.data.shape
        if len(shape) == 1:
            shape = (shape[0], 1)
        lfp_electrodes = shape[1]
        lfp_samples = shape[0]

        # Decent default rate for LFPs is 1000 Hz
        rate = 1000.0
        if lfp.rate is not None:
            rate = lfp.rate
        elif lfp.timestamps is not None:
            rate = len(lfp.timestamps) / (lfp.timestamps[-1] - lfp.timestamps[0])

        if rate < 0.1:
            rate = 1 / rate

        recording_length = lfp_samples / rate

        info = {
            "lfp_electrodes": lfp_electrodes,
            "lfp_samples": lfp_samples,
            "lfp_rate": rate,
            "lfp_recording_length": recording_length,
        }
    else:
        info = None
    return info


def largest_response_series(roi_response_series):
    """Find the largest response series in a pynwb.ophys.DfOverF object."""
    largest = None
    largest_size = 0
    for k2, v2 in roi_response_series.items():
        if isinstance(v2, pynwb.ophys.RoiResponseSeries):
            if v2.data.size > largest_size:
                largest = v2
                largest_size = v2.data.size
    return largest


def find_calcium_info(nwbfile):
    """Look for calcium imaging data in the NWB file.

    The canonical calcium imaging data is in a pynwb.ophys.RoiResponseSeries or two
    photon series, but there are many other ways to store calcium imaging data in NWB.
    """

    def look_for_calcium(acquisition, prefix=""):
        for k, v in acquisition.items():
            if isinstance(v, pynwb.ophys.RoiResponseSeries) or isinstance(
                v, pynwb.ophys.TwoPhotonSeries
            ):
                return v
            if isinstance(v, pynwb.ophys.DfOverF) or isinstance(
                v, pynwb.ophys.Fluorescence
            ):
                for k2, v2 in v.roi_response_series.items():
                    if isinstance(v2, pynwb.ophys.RoiResponseSeries):
                        return largest_response_series(v.roi_response_series)
            if isinstance(v, pynwb.base.TimeSeries) and (
                "fluo" in k.lower() or "df" in k.lower()
            ):
                # Naked time series
                return v
            elif isinstance(v, pynwb.base.ProcessingModule):
                result = look_for_calcium(v.data_interfaces, prefix=prefix + k + ".")
                if result is not None:
                    return result
        return None

    ca = look_for_calcium(nwbfile.processing, "processing.")

    if ca is None:
        ca = look_for_calcium(nwbfile.acquisition, "acquisition.")

    if ca is not None:
        shape = ca.data.shape
        ca_field_x = 0
        ca_field_y = 0
        ca_electrodes = 0

        if len(shape) == 1:
            shape = (shape[0], 1)

        if len(shape) == 3:
            ca_field_x = shape[2]
            ca_field_y = shape[1]
            ca_samples = shape[0]
        else:
            ca_electrodes = shape[1]
            ca_samples = shape[0]

        if ca_electrodes > ca_samples:
            if (ca.timestamps is not None) and (len(ca.timestamps) > ca_samples):
                # Flipped dimensions
                ca_samples, ca_electrodes = ca_electrodes, ca_samples

        # Default rate for calcium imaging is 10 Hz
        rate = 10.0
        if ca.rate is not None and ca.rate > 0:
            rate = ca.rate
        elif ca.timestamps is not None:
            rate = len(ca.timestamps) / (ca.timestamps[-1] - ca.timestamps[0])

        if rate < 0.1:
            # This is probably a mistake, assuming millisecond based timestamps.
            rate = rate * 1000.0

        recording_length = ca_samples / rate

        info = {
            "ca_rois": ca_electrodes,
            "ca_samples": ca_samples,
            "ca_field_x": ca_field_x,
            "ca_field_y": ca_field_y,
            "ca_rate": rate,
            "ca_recording_length": recording_length,
        }

    else:
        info = None
    return info


def process_dandiset(i):
    """Process a single dandiset. This involves partially downloading the NWB file and
    reading shapes to figure out the number of units, spikes, LFP channels, etc.
    Thankfully, the DANDI API allows us to do this without downloading the entire file.
    The caching filesystem is used to avoid downloading the same file multiple times.
    Over the entire archive, it nevertheless adds up to ~2.5 TB of data, so this
    takes several days to run.

    Processing a dataset and finding the right involves using a lot of heuristics to
    find relevant data–spikes, LFPs and calcium imaging.
    """
    dandiset_id = f"{i:06d}"
    print(dandiset_id)

    client = DandiAPIClient()
    try:
        dandiset = client.get_dandiset(dandiset_id)
        next(dandiset.get_assets())
    except:
        print(f"Error getting {dandiset_id}")
        return []

    m_ = dandiset.get_raw_metadata()
    try:
        data_format = m_["assetsSummary"]["dataStandard"][0]["identifier"]
    except:
        data_format = "RRID:SCR_015242"

    if data_format != "RRID:SCR_015242":
        # Only process NWB files, not BIDS.
        print(f"Skipping {dandiset_id} because it's not NWB")
        return []

    # Configure the virtual filesystem for streaming
    fs = fsspec.filesystem("http")
    fs = CachingFileSystem(
        fs=fs,
        cache_storage=f"/network/scratch/p/patrick.mineault/nwb-cache/{dandiset_id}",
    )

    datas = []

    for asset in tqdm.tqdm(dandiset.get_assets()):
        if asset.size < 1000000:
            # No point of getting tiny files.
            continue

        print(asset.identifier)
        if not asset.path.endswith("nwb"):
            continue

        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
        f = fs.open(s3_url, "rb")
        file = h5py.File(f, "r")
        io = pynwb.NWBHDF5IO(file=file, load_namespaces=True)

        try:
            nwbfile = io.read()
        except:
            print(f"Error reading {dandiset_id}")
            continue

        lfp_info = find_lfp_info(nwbfile)
        ca_info = find_calcium_info(nwbfile)
        spike_info = find_spike_info(
            nwbfile, lfp_info is not None or ca_info is not None
        )

        io.close()
        file.close()
        f.close()

        data = {"asset_id": asset.identifier, "dandiset_id": dandiset_id}

        if lfp_info is not None:
            data = {**data, **lfp_info}

        if ca_info is not None:
            data = {**data, **ca_info}

        if spike_info is not None:
            data = {**data, **spike_info}

        if "icephys" in asset.path:
            data["recording_type"] = "icephys"

        with open(f"results/dandiset-{i:03}.jsonl", "a") as f:
            dumped = json.dumps(data, default=custom_serializer)
            print(dumped)
            f.write(dumped + "\n")

        datas.append(data)
    return datas


if __name__ == "__main__":
    # All of these datasets have been found to contain lots of small files with
    # irrelevant data, so we skip them. That means microscopy data, intracellular
    # electrophysiology, patch-seq, fNIRS, etc.
    blacklist = [8, 20, 23, 26, 45, 109, 142, 209, 212, 288, 341, 489, 537, 636, 678]

    tasks = []
    for i in range(730):
        if i not in blacklist:
            tasks.append(delayed(process_dandiset)(i))

    results = Parallel(n_jobs=-1)(tasks)
