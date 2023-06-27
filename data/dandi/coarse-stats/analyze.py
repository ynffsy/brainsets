import json

ip_fname = 'dandi-metadata.json'
with open(ip_fname, 'r') as f:
    dandiset_metadata = json.loads(f.read())

i = 0
size_total = 0
for key in dandiset_metadata:

    md = dandiset_metadata[key]

    approach = md['assetsSummary']['approach']
    for a in approach:
        if 'electrophy' in a['name']:
            i += 1
            size_mb = md['assetsSummary']['numberOfBytes'] / 1024 / 1024.
            size_total += size_mb
            print(f"{i:2} {key} : {size_mb:11.3f} MB")
            break

print(f"Total size: {size_total / 1024 / 1024 : 3.3f} TB")
