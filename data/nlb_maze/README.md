There is a total of 4 sessions, each with a different size.

### Downloading the data
The data can be downloaded from the DANDI archive using the following command:
```bash
mkdir raw/ && cd $_
dandi download DANDI:000128/0.220113.0400  # Standard
dandi download DANDI:000138/0.220113.0407  # Large
dandi download DANDI:000139/0.220113.0408  # Medium
dandi download DANDI:000140/0.220113.0408 # Small
cd -
```

### Processing the data
```bash
python3 prepare_data.py
```
