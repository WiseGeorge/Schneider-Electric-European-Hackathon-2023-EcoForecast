import argparse
import pandas as pd
import numpy as np
from helper_functions import reshape_dataframe, Time_Resolutions, resample_to_1h_per_country, combine_dataframes, melt_reshape_df


def load_data(file_path1:str = 'data/Processed Data/merged_gen_data.csv',
                file_path2:str = 'dataProcessed Data/merged_load_data.csv'):
    gdf = gdf = pd.read_csv(file_path1, index_col='StartTime', parse_dates=True)
    print('Green Energy Generation DataFrame')
    print(gdf.head(5))

    ldf = pd.read_csv(file_path2, index_col='StartTime', parse_dates=True)
    print('\nLoad Energy DataFrame')
    print(ldf.head(5))

    return gdf, ldf

def clean_data(df):
    pass
    

def preprocess_data(gdf: pd.DataFrame, ldf: pd.DataFrame):

    # Reshaping GDF to Add Green Energy Tyep Columns 
    gdf = reshape_dataframe(gdf)
    # Resample DFs
    gdf = resample_to_1h_per_country(gdf)
    ldf = resample_to_1h_per_country(ldf)
    # Combine DFs
    df_processed = combine_dataframes(gdf, ldf)
    # Melt and Reshape
    fdf = melt_reshape_df(df_processed)

    return fdf

def save_data(df: pd.DataFrame, output_file:str = 'data/Processed Data/EcoForecast Dataset.csv'):
    print('\nData Processing Succefully')
    print('Full Processed DataFrame')
    print(df.head(5))
    # Save the combined DataFrame to a CSV file
    df.to_csv(output_file)
    print('DataFrame Saved to CSV File')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file1',
        type=str,
        default='data/Processed Data/merged_gen_data.csv',
        help='Path to the first raw data file to process'
    )
    parser.add_argument(
        '--input_file2',
        type=str,
        default='data/Processed Data/merged_load_data.csv',
        help='Path to the second raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/Processed Data/EcoForecast Dataset.csv', 
        help='data/procesed_data/EcoForecast Dataset.csv'
    )
    return parser.parse_args()

def main(input_file1, input_file2, output_file):
    
    ## Load Data
    gdf, ldf = load_data(input_file1, input_file2)
    ## Process Data Perform Preprocessing, Data Cleaning and Combination
    df_processed = preprocess_data(gdf, ldf)
    save_data(df_processed, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file1, args.input_file2, args.output_file)