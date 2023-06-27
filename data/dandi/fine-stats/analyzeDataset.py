from dandi import dandiapi, dandiarchive
from remoteAssetIO import remoteAssetIO
import numpy as np
import traceback
import os

def analyzeDataset(dandiID: str):

    data = {}

    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'

    url = f"https://dandiarchive.org/dandiset/{dandiID}"
    print(url)

    devices = {}
    species = set()
    num_useful_sessions = 0

    unit_sessions = [] # session ids which have nwbfile.units
    num_units = []

    trial_sessions = [] # session ids which have nwbfile.trials
    num_trials = []
    trial_times = []

    session_names = []
    session_dur_list = []
    trial_dur_list = []

    sus_session_list = []
    sus_trial_list = []

    num_spikes_list = []

    parsed_url = dandiarchive.parse_dandi_url(url)
    with parsed_url.navigate() as (client, dandiset, assets):

        # Load the most recent version
        version = dandiset.most_recent_published_version
        if version is None: # => Draft dataset
            pass
        dandiset = dandiset.for_version(version)

        ## Dandiset level metadata
        data['id'] = dandiID
        data['version'] = str(version)

        dandiset_md = dandiset.get_raw_metadata()
        data['url'] = dandiset_md['url']
        data['name'] = dandiset_md['name']
        data['num_subjects'] = dandiset_md['assetsSummary']['numberOfSubjects']

        # Number of assets seem to match number of files
        print(f"{len(list(dandiset.get_assets()))} assets")

        useful = True
        for i, asset in enumerate(assets):
            if i != 0 and useful:
                print(LINE_UP, end=LINE_CLEAR)

            asset_md = asset.get_raw_metadata()

            fname = asset_md['path']
            size_mb = asset.size / (1024.**2)
            try:
                var_meas = [x['value'] for x in asset.get_raw_metadata()['variableMeasured']]
            except KeyError:
                var_meas = []

            session_names.append(fname)

            # has Units or ElectricalSeries measurements
            useful = 'Units' in var_meas
            print(f"{i}: {fname[:40]} \t {size_mb:.3f}MB \t {useful}")

            if not useful:
                continue

            with remoteAssetIO(asset, "nwb-cache") as io:
                nwbfile = io.read()

                ## Session level metadata

                # Device
                for k in nwbfile.devices.keys():
                    if k not in devices.keys():
                        devices[k] = nwbfile.devices[k].description

                # Species
                species.add(nwbfile.subject.species)

                # Num useful sessions
                num_useful_sessions += 1

                # Num units
                nu = len(nwbfile.units)
                num_units.append(nu)
                unit_sessions.append(i)

                try:
                    session_dur = nwbfile.units.spike_times[-1] - nwbfile.units.spike_times[0]
                    session_dur_list.append(session_dur)

                    session_sus = nu * session_dur
                    sus_session_list.append(session_sus)
                except:
                    pass


                try:
                    num_spikes = len(nwbfile.units.spike_times)
                    num_spikes_list.append(num_spikes)
                except:
                    pass

                # Trials
                try:
                    num_trials.append(len(nwbfile.trials))
                    trial_str = np.array(nwbfile.trials['start_time'])
                    trial_end = np.array(nwbfile.trials['stop_time'])
                    trial_dur = trial_end - trial_str
                    trial_times.append(np.mean(trial_dur))
                    trial_sessions.append(i)

                    trial_sus = np.sum(trial_dur) * nu
                    sus_trial_list.append(trial_sus)

                    trial_dur_list.append(np.sum(trial_dur))
                except:
                    pass


    data['devices'] = devices
    data['species'] = list(species)
    data['num_useful_sessions'] = num_useful_sessions

    if num_useful_sessions > 0:
        num_units = np.array(num_units).astype(float)
        data['units'] = {}
        data['units']['session_list'] = unit_sessions
        data['units']['num_list'] = [int(x) for x in num_units]
        data['units']['mean']     = np.mean(num_units)
        data['units']['std']      = np.std(num_units)
        data['units']['max']      = np.max(num_units)
        data['units']['min']      = np.min(num_units)

        data['num_spikes'] = {}
        data['num_spikes']['session_list'] = num_spikes_list
        data['num_spikes']['total'] = sum(num_spikes_list)

        num_trials = np.array(num_trials).astype(float)
        trial_times = np.array(trial_times).astype(float)
        data['trials'] = {}
        data['trials']['session_list']  = trial_sessions
        data['trials']['num_mean']      = np.mean(num_trials)
        data['trials']['num_std']       = np.std(num_trials)
        data['trials']['duration_mean'] = np.mean(trial_times)
        data['trials']['duration_std']  = np.std(trial_times)
        data['trials']['num_list']      = [int(x) for x in num_trials]
        data['trials']['duration_list'] = list(trial_times)

        data['sus'] = {}
        data['sus']['session_total'] = np.sum(sus_session_list)
        data['sus']['trial_total']   = np.sum(sus_trial_list)
        data['sus']['session_list']  = [float(x) for x in sus_session_list]
        data['sus']['trial_list']    = [float(x) for x in sus_trial_list]

        data['session_dur_list']     = [float(x) for x in session_dur_list]
        data['total_duration']       = np.sum(session_dur_list)
        data['trial_dur_list']       = [float(x) for x in trial_dur_list]
        data['total_trial_duration'] = np.sum(trial_dur_list)

    data['sessions'] = session_names

    return data


if __name__ == "__main__":

    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dandiID', help="dandiID (eg. 000363) or filename")
    parser.add_argument('-b', '--batch', action='store_true')
    parser.add_argument('-nrw', '--norewrite', action='store_true',
                        help="Re-write existing JSONs or not")

    args = parser.parse_args()

    # Get list of dandiIDs
    if args.batch:
        fname = args.dandiID
        with open(fname, 'r') as f:
            dandiIDs = f.readlines()
    else:
        dandiIDs = [args.dandiID]

    base_data_dir = 'data/'


    # Get list of already present dataset jsons
    old_files = []
    if args.norewrite:
        old_files = os.listdir(base_data_dir)
        old_files = [x[:-5] for x in old_files if x[-5:] == '.json']


    # Analyze each dandiID
    for dandiID in dandiIDs:

        dandiID = dandiID.strip()

        # Ignore empty lines
        if len(dandiID) == 0:
            continue

        # Ignore comments
        if dandiID[0] == '#':
            print(f'Skipping {dandiID}')
            continue
        dandiID = dandiID.split('#')[0].strip()

        if dandiID in old_files:
            print(f'{dandiID}: Skipping, json already exists')
            continue

        try:
            print()
            print(dandiID)
            data = analyzeDataset(dandiID)
            print()

            fname = os.path.join(base_data_dir, dandiID + '.json')
            with open(fname, 'w') as f:
                f.write(json.dumps(data, indent=4, separators=(',', ':')))

        except:
            traceback.print_exc()
