import os
import subprocess



data_config = {
    'N1': {
        'arrays': [
            'MC', 
            'PPC', 
            None,
        ],
        'nwb_data_dir': '/home/ynffsy/Desktop/andersen_lab/data/neurogalaxy/raw/andersen_nih/N1',
        'CenterOut': [
            # '20230818',
            # '20230824',
            # '20230901',
            # '20230908',
            # '20230929',
            # '20231208',
            # '20240104', # No assist 0 run
            # '20240112', # No assist 0 run
            # '20240117', 
            # '20240119', # No assist 0 run
            # '20240126', # No assist 0 run
            # '20240201', # No assist 0 run
            # '20240202', # No assist 0 run
            # '20240208', # No assist 0 run
            # '20240209', # No assist 0 run
            # '20240214', # No assist 0 run
            # '20240215', # No assist 0 run
            # '20240222', # No assist 0 run
            # '20240223', # No assist 0 run
            # '20240229', # No assist 0 run
            # '20240306', # No assist 0 run
            # '20240313', 
            # '20240419', 
            # '20240424', 
            # '20240502', # No assist 0 run
            # '20240509', # No assist 0 run
            # '20240516', # No assist 0 run
            # '20240517', # No assist 0 run
            # '20240528', 
            # '20240529', # No assist 0 run
            # '20240605', # No assist 0 run
            # '20240606', # No assist 0 run
            # '20240607', # No assist 0 run
            # '20240612', # No assist 0 run
            # '20240613', # No assist 0 run
            # '20240614', # No assist 0 run
            # '20240627', # No assist 0 run
            # '20240705', # No assist 0 run
            # '20240719', 
            # '20240731', # No assist 0 run
            # '20240815', # No assist 0 run
            # '20240829', 
            # '20240912', # No assist 0 run
            # '20240918', # No assist 0 run
            # '20240927',
            # '20241003', # No assist 0 run
            # '20241011', # No assist 0 run
            # '20241018',
        ],
    },
    'N2': {
        'arrays': [
            'MC-LAT', 
            'MC-MED', 
            'PPC-SPL', 
            'PPC-IPL', 
            None,
        ],
        'nwb_data_dir': '/home/ynffsy/Desktop/andersen_lab/data/neurogalaxy/raw/andersen_nih/N2',
        'CenterOut': [
            '20240118',
            '20240123',
            '20240126',
            '20240130',
            '20240201',
            '20240206',
            '20240208',
            '20240213',
            '20240215',
            '20240221',
            '20240222',
            '20240228',
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
            '20240430', ## No assist 0 run
            '20240502',
            '20240503', ## No assist 0 run
            '20240509',
            '20240510', ## No assist 0 run
            '20240516',
            '20240521',
            '20240530',
            '20240618',
            '20240625', ## No assist 0 run
            '20240702',
            '20240709',
            '20240716',
            '20240816', ## No assist 0 run
            '20240820',
            '20240827',
            '20240920',
            '20240924',
            '20241001',
            '20241015',
            '20241022',
            '20241105',
            '20241106',
            '20241113',
            '20241119',
            '20241211',
            '20241212',
            '20241224',
            '20250107',
            '20250114',
            '20250128',
            '20250211',
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
