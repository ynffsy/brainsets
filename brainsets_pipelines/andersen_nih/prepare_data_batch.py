import os
import subprocess



nwb_data_dir_N1 = '/home/ynffsy/Desktop/andersen_lab/data/cg/nwb/sub-N1'
nwb_data_dir_N2 = '/home/ynffsy/Desktop/andersen_lab/data/cg/nwb/sub-N2'

nwb_sessions_N1 = {
    'CenterOut': [

    ],
    'CenterStart': [
        '20190412',
        '20190517',
        '20190528',
    ],
}

arrays_N1 = ['MC', 'PPC', None]
    
nwb_sessions_N2 = {
    'CenterOut': [

    ],
    'CenterStart': [
        '20240516',
        '20240530',
        '20240816',
        '20240820',
        '20241015',
        '20241022',
    ],
    'RadialGrid': [

    ],
    'RadialGridMasked': [

    ],
}

arrays_N2 = [
    'MC-LAT', 
    'MC-MED', 
    'PPC-SPL', 
    'PPC-IPL', 
    None]

processed_data_dir = '/home/ynffsy/Desktop/andersen_lab/data/poyo/processed/andersen_nih'



def main():

    ## Process N1 data
    for task in nwb_sessions_N1.keys():
        for session in nwb_sessions_N1[task]:
            for array in arrays_N1:
                nwb_path = os.path.join(nwb_data_dir_N1, f'sub-N1_ses-{session}_{task}.nwb')

                subprocess_call = [
                    "python", "prepare_data.py", 
                    "--input_file", nwb_path,
                    "--output_dir", processed_data_dir]
                
                if array is not None:
                    subprocess_call.extend(['--array', array])

                subprocess.run(subprocess_call)
            
    ## Process N2 data
    for task in nwb_sessions_N2.keys():
        for session in nwb_sessions_N2[task]:
            for array in arrays_N2:
                nwb_path = os.path.join(nwb_data_dir_N2, f'sub-N2_ses-{session}_{task}.nwb')

                subprocess_call = [
                    "python", "prepare_data.py", 
                    "--input_file", nwb_path,
                    "--output_dir", processed_data_dir]

                if array is not None:
                    subprocess_call.extend(['--array', array])

                # run prepare_data.py
                subprocess.run(subprocess_call)


if __name__ == "__main__":
    main()
