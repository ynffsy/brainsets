"""Load data, processes it."""
import os
import logging
import torch
import torch
from sklearn.model_selection import train_test_split
from pynwb import NWBHDF5IO
import numpy as np

from kirby.utils import find_files_by_extension, make_directory
from kirby.data import Data, IrregularTimeSeries, Interval


logging.basicConfig(level=logging.INFO)



def load_file(file_path):
    #Read the nwb file 
    io = NWBHDF5IO(str(file_path), mode='r')
    nwbfile = io.read()

    # trial table
    trials = extract_trials(nwbfile)

    # unit activity
    spikes, units = extract_spikes(nwbfile)

    data = Data(spikes=spikes, trials=trials, units=units,)
    return data


def extract_spikes(nwbfile):
    spikes = []
    unit_ids = []
    unit_meta = []

    # Get Electrodes
    try:
        channels = np.asarray(nwbfile.units['electrodes'].target.data)
    except:
        channels = np.asarray(nwbfile.units['electrodes'].data)
    
    #Get Channel IDs
    channel_ids = np.asarray(nwbfile.electrodes['origChannel'].data)
    channel_ids = channel_ids[channels]
    #Get Cluster IDs
    clusterIDs = np.asarray(nwbfile.units['origClusterID'].data).astype(int)

    for channel in range(len(channels)):
        spike_timestamps = (np.asarray(nwbfile.units.get_unit_spike_times(channel)))
        cell_id = channels[channel]

        assert isinstance(cell_id, np.number), 'expected an integer, got {}'.format(cell_id)

        spikes.append(spike_timestamps)
        unit_ids.append(channel * np.ones(len(spike_timestamps), dtype=np.int64))
        unit_meta.append((channel, clusterIDs[channel], cell_id))
    
    spikes = np.concatenate(spikes)
    unit_ids = np.concatenate(unit_ids)

    unit_meta = dict(zip(('id', 'channel_id', 'cell_id') , zip(*unit_meta)))
    for key in unit_meta:
        if np.issubdtype(type(unit_meta[key][0]), np.number):
            unit_meta[key] = torch.tensor(unit_meta[key])

    sorted = np.argsort(spikes)
    spikes = spikes[sorted]
    unit_ids = unit_ids[sorted]

    spikes = IrregularTimeSeries(torch.tensor(spikes), unit_id=torch.tensor(unit_ids))
    units = Data(**unit_meta)
    return spikes, units

def extract_response_from_nwbfile(nwbfile):
    """
    Extract recognition/learning responses from the nwbfile
    """
    experiment_description = nwbfile.experiment_description
    experiment_ids = np.unique(nwbfile.acquisition['experiment_ids'].data)
    experiment_id_recog = int(experiment_ids[1])
    experiment_id_learn = int(experiment_ids[0])

    events = (np.asarray(nwbfile.get_acquisition('events').data)).astype(float)
    experiments = np.asarray(nwbfile.get_acquisition('experiment_ids').data)

    events_recog = events[((experiments == experiment_id_recog) & ((events >= 30) & (events <= 36)))] - 30
    events_learn = events[((experiments == experiment_id_learn) & ((events >= 20) & (events <= 21)))] - 20

    return events_recog, events_learn


def extract_trials(nwbfile):
    session_id = nwbfile.identifier

    response_recog, response_learn = extract_response_from_nwbfile(nwbfile)

    stim_on = np.asarray(nwbfile.trials['start_time'].data) 
    stim_off = np.asarray(nwbfile.trials['stop_time'].data) 
    new_old_labels_recog = np.asarray(nwbfile.trials['new_old_labels_recog'].data)
    new_old_labels_recog[new_old_labels_recog == b'NA'] = b'0'
    new_old_labels_recog = new_old_labels_recog.astype(int)

    category_id = np.asarray(nwbfile.trials['stimCategory'].data).astype(int)
    delay1_off = np.asarray(nwbfile.trials['delay1_time'].data)
    delay2_off = np.asarray(nwbfile.trials['delay2_time'].data)
    category_name = np.asarray((nwbfile.trials['category_name'].data))
    stim_phase = np.asarray(nwbfile.trials['stim_phase'].data)
    stim_phase_id = np.zeros(len(stim_phase), dtype=int)
    stim_phase_id[stim_phase == 'recog'] = 1

    response_time = np.asarray(nwbfile.trials['response_time'].data) 

    trials = Interval(start = torch.tensor(stim_on) - 1.0,
                      end = torch.tensor(stim_on) + 4.0,
                      stim_on=torch.tensor(stim_on),
                      stim_off=torch.tensor(stim_off),
                      delay1_off=torch.tensor(delay1_off),
                      delay2_off=torch.tensor(delay2_off),
                      # labels
                      new_old_labels_recog=torch.tensor(new_old_labels_recog),
                      category_id=torch.tensor(category_id),
                      category_name=category_name,
                      stim_phase=stim_phase,
                      stim_phase_id=torch.tensor(stim_phase_id),
                      response=torch.tensor(np.concatenate([response_learn, response_recog])),
                      response_time=torch.tensor(response_time),
                    )
    return trials

#########
# Split #
#########


def split_and_get_train_validation_test(trials, test_size=0.2, valid_size=0.1, random_state=42):
    assert 0 < valid_size < 1, "valid_size must be positive, got {}".format(valid_size)
    assert 0 < test_size < 1, "test_size must be positive, got {}".format(test_size)

    num_trials = len(trials)
    train_size = 1. - test_size - valid_size
    assert 0 < train_size < 1, "train_size must be positive, got {}".format(train_size)
    
    train_valid_ids, test_ids = train_test_split(np.arange(num_trials), test_size=test_size, random_state=random_state)
    train_ids, valid_ids = train_test_split(train_valid_ids, test_size=valid_size/(train_size+valid_size), random_state=random_state)
    
    train_trials = [trials[i] for i in train_ids]
    valid_trials = [trials[i] for i in valid_ids]
    test_trials = [trials[i] for i in test_ids]

    return train_trials, valid_trials, test_trials


def collect_slices(data, trials):
    slices = []
    for i, trial in enumerate(trials):
        start, end = trial['start'], trial['end']
        data_slice = data.slice(start, end)
        data_slice.trials = data.trials[i]
        slices.append(data_slice)
    return slices


if __name__ == "__main__":
    raw_folder_path = "./raw"
    processed_folder_path = "./processed"
    make_directory(processed_folder_path, prompt_if_exists=True)
    make_directory(os.path.join(processed_folder_path, 'train'))
    make_directory(os.path.join(processed_folder_path, 'valid'))
    make_directory(os.path.join(processed_folder_path, 'test'))

    extension = ".nwb"
    session_list = []
    # find all files with extension .nwb in folder_path
    for file_path in find_files_by_extension(raw_folder_path, extension):
        logging.info(f"Processing file: {file_path}")
        data = load_file(file_path)

        train_trials, validation_trials, test_trials = split_and_get_train_validation_test(data.trials, test_size=0.2, valid_size=0.1, 
                                                                       random_state=42)

        # collect data slices for validation and test trials
        train_slices = collect_slices(data, train_trials)
        validation_slices = collect_slices(data, validation_trials)
        test_slices = collect_slices(data, test_trials)


        # for each bucket we estimate the expected number of input tokens
        # for bucket in train_buckets:
        #     max_num_input_tokens = get_num_input_tokens(train_buckets[1], WINDOW_SIZE, JITTER_PADDING)  # 3233, 3249, 3249, 3265, 3271, 3272, 3273, 3283, 3296, 3353
        #     batch_bucket_size = next_power_of_2(max_num_input_tokens)
        #     bucket.max_num_input_tokens = max_num_input_tokens
        #     bucket.batch_bucket_size = batch_bucket_size

        # all files are saved in their corresponding folders

        for i, sample in enumerate(train_slices):
            zid = str(i).zfill(5)
            filename = os.path.splitext(os.path.basename(file_path))[0] + f'_{zid}.pt'
            path = os.path.join(processed_folder_path, 'train', filename)
            torch.save(sample, path)

        count = len(train_slices)
        for i, sample in enumerate(validation_slices):
            zid = str(i + count).zfill(5)
            filename = os.path.splitext(os.path.basename(file_path))[0] + f'_{zid}.pt'
            path = os.path.join(processed_folder_path, 'valid', filename)
            torch.save(sample, path)

        count += len(validation_slices)
        for i, sample in enumerate(test_slices):
            zid = str(i + count).zfill(5)
            filename = os.path.splitext(os.path.basename(file_path))[0] + f'_{zid}.pt'
            path = os.path.join(processed_folder_path, 'test', filename)
            torch.save(sample, path)

        session_id = os.path.splitext(os.path.basename(file_path))[0]
        num_units = data.spikes.unit_id.max() + 1
        session_list.append((session_id, num_units))

    # save session_list as txt
    info_path = "all.txt"
    with open(info_path, 'w') as f:
        for session_id, num_units in session_list:
            f.write(f'{session_id} {num_units}\n')
