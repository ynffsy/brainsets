""" getIds.py

Print out all valid DANDI Dataset IDs

Valid: Must have 'Units' as a variable
"""

from dandi.dandiapi import DandiAPIClient

with DandiAPIClient.for_dandi_instance("dandi") as client:
    for dandiset in client.get_dandisets():

        md = dandiset.get_raw_metadata()

        # Get ID
        idstr = md['id'].split(':')[1].split('/')[0]

        try:
            var_meas = [x.lower() for x in md['assetsSummary']['variableMeasured']]
        except:
            print(f"# {idstr} # Missing assetsSummary")
            continue

        if 'units' not in var_meas:
            print(f"# {idstr} # No Units")
            continue

        if dandiset.most_recent_published_version is None:
            print(f"  {idstr} # DRAFT")
            continue

        print(f"  {idstr}")
