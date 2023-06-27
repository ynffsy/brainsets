### Downloading the data
The data can be downloaded from the DANDI archive using the following command:
```bash
mkdir raw/ && cd $_
dandi download https://dandiarchive.org/dandiset/000129/draft
```

### Processing the data
```bash
python3 prepare_data.py
```

