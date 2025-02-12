import os
import subprocess



data_config = {
    'N1': {
        'arrays': [
            # 'MC', 
            'PPC', 
            None,
        ],
        'nwb_data_dir': '/home/ynffsy/Desktop/andersen_lab/data/neurogalaxy/raw/andersen_nih/N1',
        'CenterOut': [
            # '20230726',
            '20230818',
            '20230824',
            '20230901',
            '20230908',
            '20230929',
            '20231208',
            '20240912',
            '20240918',
            '20240927',
            '20241003',
            '20241011',
            '20241018',
        ],
    },
    'N2': {
        'arrays': [
            # 'MC-LAT', 
            'MC-MED', 
            'PPC-SPL', 
            'PPC-IPL', 
            None,
        ],
        'nwb_data_dir': '/home/ynffsy/Desktop/andersen_lab/data/neurogalaxy/raw/andersen_nih/N2',
        'CenterOut': [
            '20240222',
            '20240229',
            '20240306',
            '20240308',
            '20240312',
            '20240314',
            '20240319',
            '20240328',
            '20240402',
            '20240409',
            '20240411',
            '20240418',
            '20240430',
            '20240502',
            '20240503',
            '20240509',
            '20240510',
            '20240516',
            '20240521',
            '20240530',
            '20240618',
            '20240625',
            '20240702',
            '20240709',
            '20240716',
            '20240816',
            '20240820',
            '20240827',
            '20240920',
            '20240924',
            '20241001',
            '20241015',
            '20241022',
            '20241105',
            '20241211',
        ]
    }
}

processed_data_dir = '/home/ynffsy/Desktop/andersen_lab/data/neurogalaxy/processed/andersen_nih'



def main():

    for subject in data_config.keys():

        arrays = data_config[subject]['arrays']
        nwb_data_dir = data_config[subject]['nwb_data_dir']
        session_dates = data_config[subject]['CenterOut']

        for session in session_dates:
            for array in arrays:
                nwb_path = os.path.join(nwb_data_dir, f'sub-{subject}_ses-{session}_CenterOut.nwb')

                subprocess_call = [
                    "python", "prepare_data.py", 
                    "--input_file", nwb_path,
                    "--output_dir", processed_data_dir]
                
                if array is not None:
                    subprocess_call.extend(['--array', array])

                subprocess.run(subprocess_call)



if __name__ == "__main__":
    main()
