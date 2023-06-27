# Neural datasets 
Each folder contains the scripts used to download and pre-process a dataset. The folder name is either the last names of the first and last authors of the publication that accompanies the released dataset, or the name of the benchmark. Each dataset folder contains a `README.md` file that describes the data, and how to use it. 

**#todo** create a template for a dataset card, that includes details about pre-processing, stats about the data (total number of sessions, number of neurons, file size, availbale behavioral variables, tasks, etc.) 

**#todo** important to include the license for each dataset. 

## Downloading the data
Links or scripts to download the data should be included in the `README.md` file. Private dataset (like Matt's) are currently available here: [OneDrive](https://gtvault-my.sharepoint.com/:f:/r/personal/mazabou3_gatech_edu/Documents/Project%20Kirby?csf=1&web=1&e=MZDWEW) (send request to Mehdi for access), and we can plan to consolidate all the data in a single location internally (and publically?) in the future.

If you are using the Dyer Lab clusters, all data is mounted in `/kirby`. 

Currently the following datasets are available:
| Folder | Status | Description |
| --- | --- | --- | 
| Perich, Miller | 
| Philip, Makin | 
| Churchland, Shenoy |
| Flint, Slutzky | 
| Chandravadia, Rutishauser |
| nlb_maze | 
| nlb_rtt | 
| allen v1 |

## Processing the data

Currently the processing scripts are designed to:
- split the data into train/test/validation sets
- extract neural data and behavioral targets
- clean data to remove outliers (e.g. high acceleration in Matt's data)
- normalize the data (e.g. z-score neural data)
- chunk the data into random windows of a fixed size (e.g. 1s), or into trials (usually for evaluation). Note that under the hood, we use a window of size 2s, and use a step of 1s to create overlapping windows. Then during training we randomly crop the time window to 1s. 

**#todo** the choice of splitting the data into chunks is not ideal because it depends on a window size. The challenge is in that we have irregular timeseries (spikes), each time we want to clip a sequence between t1 and t2 we need to find the indices of the closest timepoints in the data. Currently we make sure that the spikes are sorted and that the timestamps are monotonically increasing, so we can use `np.searchsorted` to find the indices of the closest timepoints. 
Look into:
- some sort of pre-computed KD tree
- `np.memmap`

There are three "types of objects":

| Object | Description | Useful for |
| --- | --- | --- |
| IrregularTimeseries | first attribute is `timestamps` of size `(L)`, and then any other attributes `x`, `y` of size `(L, *)` | e.g. spikes, behavior |
| Interval | had `start` and `end`| e.g. trials, windows |
| RegularTimeseries | timestamps do not need to be defined, just the sampling frequency, should be easier to slice | currently not used |

**#todo** the current way we split the data into train/test might be arbitrary? something to discuss in regards to building a benchmark.

Once the data is processed, all the files are saved to one of three folders `train`, `test`, `valid`. 

## Loading the datasets
The dataset can be called as follows:
```python
train_dataset = Dataset('./data/', split='train',
                        include=[
                            "Perich, Miller/MrT_RT_FF_BL_08202013_001_stripped"
                            "Perich, Miller/chewie", 
                            "Perich, Miller/mihili",
                            "Flint, Slutzky/*",
                            ], 
                        transform=transform)
```

Any number of datasets can be included. First, a specific session is including by specifying the folder name, followed by the session id. Second, a group of sessions from a datasets is included by specifying the folder name followed by a `/` and the group id. There should exist a text file `[group_id].txt` in the dataset folder that lists the sessions to include. Third, a whole dataset can be included by specifying the folder name followed by `/*`, note that `*` itself is a group id corresponding to `all` sessions (i.e. there is a file names `all.txt`).
