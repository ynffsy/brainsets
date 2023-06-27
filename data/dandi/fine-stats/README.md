# Fine-stats

- `python analyzeDataset.py [-h] <dandiID or -b batchFile> [-nrw]`: Outputs in `data/*.json`
  - Example: `python analyzeDataset.py -b allIds.txt -nrw`
- `python getIds.py > file`
- `remoteAssetIO.py`: Context manager for remote HDF5 asset IO
- `allIds.txt`: Curated batchfile for all dandiIds to be scraped for stats
- `ephys.txt`: Curated batchfile of Ephys dandisets
- `consolidate.py`: Helper script to consolidate some fields from all dataset
- `data/`: The data directory with json files
