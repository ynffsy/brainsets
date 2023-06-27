# A NWB-based Dataset and Processing Pipeline of Human Single-Neuron Activity During a Declarative Memory Task

## Description
- Author: Ueli Rutishauser
- Download link: https://osf.io/hv7ja/
- Code: https://github.com/rutishauserlab/recogmem-release-NWB

Notes from Ueli:
> The 2015 paper describes the task and scientific results, the 2020 Scientific Data is the data release and described structure of the files etc. 

> To get started with this dataset, I would first focus on plotting some of the VS cells and compare them with the plots shown in the scientific data paper to make sure its identical.

> The Minxha 2020 is the other task we described, which also has the choice coding and preSMA/dACC cells. That data is here: https://osf.io/u3kcp/   . the “new/old” subtask of this experiment is identical to the one in the Chandravadia paper and there are also VS/MS cells in the MTL, so this could be a dataset to do “across task training” if that becomes an interest.


## Downloading the data
The data is public available, can be downloaded here: https://osf.io/hv7ja/


## Analysis Code
Analysis code is available, pulling the repo from github (requires `pynwb`):
```bash
git clone https://github.com/rutishauserlab/recogmem-release-NWB.git
```

Analysis is run as follows:
```python 
import RutishauserLabtoNWB as RLab

NWBFilePath = 'V:\\LabUsers\\chandravadian\\NWB Data\\python'
list_of_patients_behavior = [5, 6]  # List of sessions to summarize behavior. Set to [] to skip
list_of_patients_neurons = [132]    # List of sessions to analyze neural data for. Set to [] to skip

RLab.NO2NWB_analysis(NWBFilePath,list_of_patients_behavior, list_of_patients_neurons)

```

* **NWBFilePath**: The path to the exported NWB files. 
* **list_of_patients_behavior**: This signifies the NOID (Patient Identifier). See defineNOsessions.ini to see the NOID for all patients. Note: You can list more than one NOID. For example, ```list_of_patients = [5, 6, 132]```. You can also input ```list_of_patients = 'all'``` to examine the results for all patients. 
* **list_of_patients_neurons**: This signifies the NOID (Patient Identifier). See defineNOsessions.ini to see the NOID for all patients. Note: You can list more than one NOID. For example, ```list_of_patients = [5, 6, 132]```

## Preparing the data for Kirby
