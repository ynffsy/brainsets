import contextlib
from fsspec.implementations.cached import CachingFileSystem
import h5py
import fsspec
from pynwb import NWBHDF5IO
from dandi import dandiapi


@contextlib.contextmanager
def remoteAssetIO(asset: dandiapi.RemoteAsset, cache_path: str) -> NWBHDF5IO:
    """Open a Dandi RemoteAsset as an NWB IO object with file caching
    """
    fs = CachingFileSystem(
        fs=fsspec.filesystem("http"),
        cache_storage=cache_path
    )

    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

    with fs.open(s3_url, "rb") as f:
        with h5py.File(f) as file:
            with NWBHDF5IO(file=file, load_namespaces=True) as io:
                yield io
