### Downloading the data

The data can be downloaded from the DANDI archive using the following command:
```bash
mkdir raw/ && cd $_
dandi download DANDI:000121/0.220124.2156
cd -
```

### Processing the data
```bash
python3 prepare_data.py
```
