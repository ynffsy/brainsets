import numpy as np
from kirby.data.signal import downsample_wideband, extract_bands


def test_extract_bands():
    bands, ts, band_names = extract_bands(
        np.random.randn(1000, 10), np.arange(1000), Fs=1000, notch=60
    )
    assert bands.ndim == 3
    assert ts.size == bands.shape[0]
    assert bands.shape[2] == len(band_names)
