import pandas as pd
import numpy as np
import os

def merge_load_data():
    # List all csv files in the data directory
    files = [f for f in os.listdir('data/Raw Data') if f.startswith('load_') and f.endswith('.csv')]

    # Initialize an empty DataFrame to store all data
    all_data = pd.DataFrame()

    for file in files:
        try:
            # Load the data
            data = pd.read_csv(f'data/Raw Data/{file}')

            # Convert StartTime to datetime and set as index
            data['StartTime'] = pd.to_datetime(data['StartTime'], format='%Y-%m-%dT%H:%M%zZ')
            data.set_index('StartTime', inplace=True)

            # Identify the country from the file name
            country = file.split('_')[1].split('.')[0]  # Remove file extension

            # Add a 'Country' column with the country code
            data['Country'] = country

            # Append the data to the all_data DataFrame
            all_data = all_data.append(data)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Save the merged data to a new CSV file
    all_data.to_csv('data/Processed Data/merged_load_data.csv')

    print("Data merging completed. The merged data is saved as 'merged_load_data.csv'.")

def merge_gen_data():
    # Define a dictionary to map PsrType codes to energy types
    psr_type_dict = {
        'A03': 'Mixed',
        'A04': 'Generation',
        'A05': 'Load',
        'B01': 'Biomass',
        'B02': 'Fossil Brown coal/Lignite',
        'B03': 'Fossil Coal-derived gas',
        'B04': 'Fossil Gas',
        'B05': 'Fossil Hard coal',
        'B06': 'Fossil Oil',
        'B07': 'Fossil Oil shale',
        'B08': 'Fossil Peat',
        'B09': 'Geothermal',
        'B10': 'Hydro Pumped Storage',
        'B11': 'Hydro Run-of-river and poundage',
        'B12': 'Hydro Water Reservoir',
        'B13': 'Marine',
        'B14': 'Nuclear',
        'B15': 'Other renewable',
        'B16': 'Solar',
        'B17': 'Waste',
        'B18': 'Wind Offshore',
        'B19': 'Wind Onshore',
        'B20': 'Other',
        'B21': 'AC Link',
        'B22': 'DC Link',
        'B23': 'Substation',
        'B24': 'Transformer'
    }

    # List all csv files in the data directory
    files = [f for f in os.listdir('data/Raw Data') if f.startswith('gen_') and f.endswith('.csv')]

    # Initialize an empty DataFrame to store all data
    all_data = pd.DataFrame()

    for file in files:
        # Identify the country and PsrType from the file name
        split_name = file.split('_')
        country = split_name[1]
        psr_type = split_name[2].split('.')[0]  # Remove file extension

        # Skip files with PsrType codes not in the provided dictionary
        if psr_type not in psr_type_dict:
            continue
        try:    
            # Load the data
            data = pd.read_csv(f'data/Raw Data/{file}')

            # Convert StartTime to datetime and set as index
            data['StartTime'] = pd.to_datetime(data['StartTime'], format='%Y-%m-%dT%H:%M%zZ')
            data.set_index('StartTime', inplace=True)

            # Replace PsrType codes with energy types
            data['PsrType'] = psr_type_dict[psr_type]

            # Add a 'Country' column with the country code
            data['Country'] = country

            # Append the data to the all_data DataFrame
            all_data = all_data.append(data)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Save the merged data to a new CSV file
    all_data.to_csv('data/Processed Data/merged_gen_data.csv')

    print("Data merging completed. The merged data is saved as 'merged_gen_data.csv'.")

def Start_ETL():
    merge_load_data()
    merge_gen_data()


if __name__ == "__main__":
    Start_ETL()