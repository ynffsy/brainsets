import numpy as np
from kirby.data.signal import downsample_wideband, extract_bands, cube_to_long
import torch

def test_extract_bands():
    bands, ts, band_names = extract_bands(
        np.random.randn(1000, 10), np.arange(1000), Fs=1000, notch=60
    )
    assert bands.ndim == 3
    assert ts.size == bands.shape[0]
    assert bands.shape[2] == len(band_names)

def test_cube_to_long():
    ts = np.arange(10)
    cube = np.zeros((1, 10, 2), dtype=int)
    cube[0, 6, 0] = 2
    cube[0, 5, 1] = 3

    trials, units = cube_to_long(ts, cube)

    assert len(units.unit_name) == 2
    assert len(trials) == 1
    assert len(trials[0].timestamps) == 5
    assert torch.all(torch.tensor([1, 1, 1, 0, 0]).eq(torch.tensor(trials[0].unit_index)))
    assert torch.all(torch.tensor([5, 5, 5, 6, 6]).eq(torch.tensor(trials[0].timestamps)))
