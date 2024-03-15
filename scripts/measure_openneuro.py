import io
import json
import os
import subprocess
import warnings

import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from tenacity import retry, wait_exponential

# Constants
ENDPOINT = "https://openneuro.org/crn/graphql"
MRI_MODALITY = "MRI"
EEG_LIKE_MODALITIES = ("EEG", "MEG", "iEEG")


initial_query = """
query ($after: String) {
  datasets(after: $after) {
    edges {
      node {
        metadata {
          modalities
          datasetId
          datasetName
          species
          seniorAuthor
          studyDesign
          studyDomain
        }
        latestSnapshot {
          tag
        }
        name
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
  participantCount
}
"""


snapshot_query = """
query snapshotFiles($datasetId: ID!, $tree: String, $tag: String!) {
    snapshot(datasetId: $datasetId, tag: $tag) {
        files(tree: $tree) {
            id
            key
            filename
            size
            directory
            annexed
            urls
        }
    }
}
"""


def custom_serializer(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def graphql_query(query, variables=None):
    """
    I had to resort to using a closure because @retry in the root doesn't play well with
    joblib (complains about pickling).
    """

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    def _graphql_query(query, variables=None):
        response = requests.post(
            ENDPOINT, json={"query": query, "variables": variables}
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Query failed with status code {response.status_code}")

    return _graphql_query(query, variables)


# Function to recursively fetch all datasets with a relevant modality
def fetch_relevant_datasets():
    relevant_datasets = []
    has_next_page = True
    cursor = None

    all_modalities = set([MRI_MODALITY]).union(set(EEG_LIKE_MODALITIES))

    while has_next_page:
        print(f"Fetching page with cursor {cursor}")
        variables = {"after": cursor} if cursor else {}
        response = graphql_query(initial_query, variables)
        datasets = response["data"]["datasets"]["edges"]

        # Add datasets with MRI modality to the list
        relevant_datasets += [
            node
            for node in datasets
            if (
                set(node["node"]["metadata"]["modalities"]).intersection(all_modalities)
            )
        ]

        # Update cursor and check for next page
        page_info = response["data"]["datasets"]["pageInfo"]
        has_next_page = page_info["hasNextPage"]
        cursor = page_info["endCursor"] if has_next_page else None

    return relevant_datasets


def extract_timebase(data):
    if isinstance(data, dict):
        if "Units" in data:
            return data["Units"]
        else:
            return "s"
    else:
        # Actually a string
        if "milli" in data:
            return "ms"
        else:
            return "s"


def find_subjects(dataset_id, tag):
    files_response = graphql_query(
        snapshot_query, {"datasetId": dataset_id, "tag": tag}
    )

    # Select all the directories within
    if files_response is None:
        return []

    subjects = [
        file
        for file in files_response["data"]["snapshot"]["files"]
        if file["directory"] and file["filename"].startswith("sub-")
    ]

    # Now find units
    event_file = [
        file
        for file in files_response["data"]["snapshot"]["files"]
        if not file["directory"] and file["filename"].endswith("events.json")
    ]

    if event_file and len(event_file) == 1:
        # Load the events file to understand the units
        try:
            event_response = requests.get(event_file[0]["urls"][0])
        except requests.exceptions.ConnectionError:
            warnings.warn(
                f"Could not connect to remote file {event_file[0]['filename']}"
            )
            return subjects, None

        event_metadata = json.loads(event_response.content.decode("utf-8"))
        try:
            onset_unit = extract_timebase(event_metadata.get("onset", {}))
            duration_unit = extract_timebase(event_metadata.get("duration", {}))

            multipliers = {
                "s": 1,
                "second": 1,
                "seconds": 1,
                "ms": 1e-3,
                "millisecond": 1e-3,
                "milliseconds": 1e-3,
            }

            return subjects, (
                multipliers.get(onset_unit, 1),
                multipliers.get(duration_unit, 1),
            )
        except (KeyError, AttributeError):
            return subjects, None

    return subjects, None


def find_target_dir(root, dataset_id, tree, tag, target="func"):
    """
    Find a target directory within a root directory.

    Params:
        root: the root directory to search
        dataset_id: the dataset id
        tree: the tree id
        tag: the tag of the snapshot of the dataset
        target: the target directory to search for
    """
    files_response = graphql_query(
        snapshot_query, {"datasetId": dataset_id, "tree": tree, "tag": tag}
    )

    # Select all the directories within
    func_dirs = [
        file
        for file in files_response["data"]["snapshot"]["files"]
        if file["directory"] and file["filename"] == target
    ]
    if len(func_dirs) == 1:
        return [(root + "/func", func_dirs[0])]
    else:
        sess_dirs = [
            file
            for file in files_response["data"]["snapshot"]["files"]
            if file["directory"] and file["filename"].startswith("ses-")
        ]
        if sess_dirs:
            return sum(
                [
                    find_target_dir(
                        root + "/" + sess_dir["filename"],
                        dataset_id,
                        sess_dir["id"],
                        tag,
                        target,
                    )
                    for sess_dir in sess_dirs
                ],
                [],
            )
        else:
            return []


def get_duration_from_tsv(events_files, type="task-fmri", units=None):
    """
    This gets the duration of an experiment by looking at the times between the start
    and end of the events files. It doesn't work for resting state scans. It can
    sometimes break when units other than seconds are used, hence the units argument.
    """
    total_duration = 0
    for events_file in events_files:
        try:
            events_response = requests.get(events_file["urls"][0])
        except requests.exceptions.ConnectionError:
            warnings.warn(f"Could not connect to remote file {events_file['filename']}")
            continue

        # Load this as a pandas dataframe after casting the bytes to a string
        try:
            events_df = pd.read_csv(
                io.StringIO(events_response.content.decode("utf-8")), sep="\t"
            )
        except UnicodeDecodeError:
            warnings.warn(f"Empty events file {events_file['filename']}")
            continue

        if units is None:
            units = (1, 1)

        onset_mult = units[0]
        duration_mult = units[1]

        try:
            first_event = events_df.iloc[0]
            last_event = events_df.iloc[-1]

            try:
                events_df["duration"] = events_df["duration"].fillna(0)
                last_event_duration = float(last_event["duration"])
            except (KeyError, IndexError, TypeError, ValueError):
                last_event_duration = 0

            # Important to remove the first event offset because some events files start
            # with a non-zero timestamp.
            total_duration += (
                last_event["onset"] - first_event["onset"]
            ) * onset_mult + last_event_duration * duration_mult
        except (KeyError, IndexError, TypeError) as e:
            print(e)
            warnings.warn(f"Empty events file {events_file['filename']}")
    return {"total_duration": total_duration, "type": type}


def get_size_from_extrapolation(root, relevant_files):
    """
    This calculates the size by extrapolating from the size of a single scan file.
    This is the solution proposed by
    https://neurostars.org/t/total-amount-of-fmri-hours-on-openneuro/27509/7
    Unfortunately it's a pain to get datalad working properly in a subprocess, so
    I gave up on this.
    """
    multiplier = sum([x["size"] for x in relevant_files]) / relevant_files[0]["size"]

    # Get the number of frames of the first file
    cmd1 = [
        "datalad",
        "fsspec-head",
        "-c",
        "1024",
        f"{root}/{relevant_files[0]['filename']}",
    ]
    cmd2 = [
        "python",
        os.path.expanduser("~/Documents/project-kirby/scripts/read_len.py"),
    ]

    print("calling subprocess with command: ")
    print(" ".join(cmd1))

    # Run the first command and capture its output
    process1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, cwd="openneuro/openneuro")

    # Run the second command using the output of the first command as its input
    process2 = subprocess.Popen(cmd2, stdin=process1.stdout, stdout=subprocess.PIPE)

    # Close the output of the first process to allow it to receive a SIGPIPE if the
    # second process exits before it.
    process1.stdout.close()

    # Get the final output
    output, _ = process2.communicate()

    # Convert the output to an integer
    duration = int(output)

    # Calculate the total size of the directory
    total_duration = duration * multiplier

    return {"total_duration": total_duration, "type": "rs-fmri"}


def estimate_directory_size(dataset_id, tree, tag, units, data_type):
    """
    Estimate the total size of a directory by summing the documented size of all the
    tsv event files.
    """
    files_response = graphql_query(
        snapshot_query, {"datasetId": dataset_id, "tree": tree, "tag": tag}
    )

    # Select all the files either finishing with _bold.nii.gz or _events.tsv
    data_files = {}
    events_files = {}
    if files_response["data"]["snapshot"]["files"] is None:
        return {"total_duration": 0, "type": "rs-fmri", "nbytes": 0, "nfiles": 0}

    endings = ["_bold.nii.gz", "_bold.nii", "_bold.nii.gz"]

    for file in files_response["data"]["snapshot"]["files"]:
        if any(file["filename"].endswith(x) for x in endings):
            begin = file["filename"].split("gz")[0]
            data_files[begin] = file
        elif file["filename"].endswith("_events.tsv"):
            begin = file["filename"].split("_events.tsv")[0]
            events_files[begin] = file

    # Check that each bold file has a corresponding events file
    nbytes = sum([x["size"] for x in files_response["data"]["snapshot"]["files"]])
    nfiles = len(data_files)
    pairs = set(data_files.keys()).intersection(set(events_files.keys()))

    if data_type.lower() == "fmri" and len(pairs) == 0:
        # This is likely a resting state scan.
        # Read the first one in the directory, then extrapolate to the total number in
        # the directory.
        return {
            "total_duration": 0,
            "nbytes": nbytes,
            "nfiles": nfiles,
            "type": "rs-fmri",
        }
    else:
        data = get_duration_from_tsv(events_files.values(), units=units)
        return {**data, "nbytes": nbytes, "nfiles": nfiles}


def get_one_dataset(dataset_id, tag, modalities):
    print(f"Processing {dataset_id} {tag}")
    f = open(f"results/openneuro-{dataset_id}.jsonl", "a")

    subjects, units = find_subjects(dataset_id, tag)
    for subject in subjects:
        for modality in modalities:
            total_duration = 0
            total_size = 0
            nfiles = 0
            data_type = ""

            func_dirs = find_target_dir(
                f"{dataset_id}/{subject['filename']}",
                dataset_id,
                subject["id"],
                tag,
                "func",
            )

            for _, func_dir in func_dirs:
                results = estimate_directory_size(
                    dataset_id, func_dir["id"], tag, units, modality
                )
                total_size += results["nbytes"]
                total_duration += results["total_duration"]
                nfiles += results["nfiles"]
                data_type = results["type"]

            json.dump(
                {
                    "dataset_id": dataset_id,
                    "subject": subject["filename"],
                    "total_duration": total_duration,
                    "bytes": total_size,
                    "data_type": data_type,
                    "files": nfiles,
                },
                f,
                default=custom_serializer,
            )
        f.write("\n")
    f.close()


if __name__ == "__main__":
    # Fetch MRI datasets
    mri_datasets = fetch_relevant_datasets()
    tasks = []
    for dataset in mri_datasets:
        dataset_id = dataset["node"]["metadata"]["datasetId"]
        tag = dataset["node"]["latestSnapshot"]["tag"]
        modalities = dataset["node"]["metadata"]["modalities"]

        tasks.append(delayed(get_one_dataset)(dataset_id, tag, modalities))

    results = Parallel(n_jobs=-1)(tasks)
