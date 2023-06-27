import os
import json
import numpy as np

base_dir = 'data/'

files = os.listdir(base_dir)
files.sort()

for fname in files:
    if fname[-5:] != '.json':
        continue

    with open(os.path.join(base_dir, fname), 'r') as f:
        data = json.loads(f.read())

    if not data['num_useful_sessions'] > 0:
        continue

    # IDS
    #print(data['id'])

    # URL
    #print(data['url'])

    # Name
    #print(data['name'])

    # Number of useful sessions
    #print(data['num_useful_sessions'])

    # Number of subjects
    #print(data['num_subjects'])

    # Species
    #print(', '.join([str(x) for x in data['species']]))

    # Draft or not
    #print(int(data['version'] == "None"))
    #print('Draft' if data['version'] == "None" else 'Published')

    # Number of spikes
    #print(f"{data['id']} {data['num_spikes']['total'] / (1024**2.):.3f}")
    try:
        print(f"{data['id']} {data['num_spikes']['total'] / (1024**2.):.3f}")
    except:
        print(f"{data['id']} Missing")

    # Session time
    if False:
        d = data['total_duration'] / 60. / 60.
        if d == 0.0:
            print('Missing')
        elif np.isnan(d):
            print('Error')
        else:
            print(f"{d:.3f}")

    # Trial time
    if False:
        d = data['total_trial_duration'] / 60. / 60.
        if d == 0.0:
            print('Missing')
        elif np.isnan(d):
            print('Error')
        else:
            print(f"{d:.3f}")

    # Device
    #print(data['devices'])

    # Num units
    #print(sum(data['units']['session_list']))

    # MSUS-session
    #print(f"{data['sus']['session_total'] / 1e6:.3f}")

    # MSUS-trial
    if False:
        d = data['sus']['trial_total'] / 1e6
        if d == 0.0:
            print('Missing')
        elif np.isnan(d):
            print('Error')
        else:
            print(f"{d:.3f}")
