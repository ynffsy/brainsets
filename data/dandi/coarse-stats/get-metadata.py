"""
get-metadata.py
---------------

Download metadata for all dandisets into one dictionary and save into
`dandi-metadata.json`
"""

from dandi.dandiapi import DandiAPIClient
import json

# Get metadata into one big dictionary
md = {}  # {'id' : {raw_metadata}, ...}
with DandiAPIClient.for_dandi_instance("dandi") as client:
    i = 0
    for dandiset in client.get_dandisets():
        if dandiset.most_recent_published_version is None:
            continue
        i += 1
        md[str(dandiset)] = dandiset.get_raw_metadata()
        print(str(i) + ': ' + str(dandiset))

# Save the metadata dictionary
op_fname = 'dandi-metadata.json'
with open(op_fname, 'w') as f:
    f.write(json.dumps(
        md,
        indent=4,
        separators=(',', ':')
    ))
