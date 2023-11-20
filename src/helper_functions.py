
import pandas as pd
import numpy as np

def reshape_dataframe(df):
    df.index = pd.to_datetime(df.index)

    df_pivot = df.pivot_table(index=[df.index, 'Country'], columns='PsrType', values='quantity')

    df_pivot.reset_index(level='Country', inplace=True)

    df_pivot.fillna(0.0, inplace=True)

    return df_pivot


def Time_Resolutions(df):
    
    df.index = pd.to_datetime(df.index)

    series_tiempo = {}

    for pais in df['Country'].unique():
        series_tiempo[pais] = df[df['Country'] == pais].index.to_series()

    # Calcular e imprimir las resoluciones de tiempo para cada país
    for pais, serie in series_tiempo.items():
        print(f'Country: {pais}')
        print('Time Resolutions (in minutes):')
        diff = serie.diff().dt.total_seconds() / 60
        print(diff[~np.isnan(diff)].unique())  # Eliminar los valores NaN antes de imprimir


def resample_to_1h_per_country(df):
    """
    This function resamples a DataFrame to a time resolution of 1 hour for each country separately and sums up the values of each type of energy.

    Parameters:
    df (pandas.DataFrame): The input DataFrame. It is assumed that the DataFrame's index is of type datetime and that there is a 'Country' column.

    Returns:
    df_resampled (pandas.DataFrame): The resampled DataFrame.
    """

    # Create a dictionary to store the resampled DataFrames for each country
    dfs_resampled = {}

    # For each country in the DataFrame, resample to a time resolution of 1 hour and sum the values
    for country in df['Country'].unique():
        df_country = df[df['Country'] == country]
        df_country_resampled = df_country.resample('1H').sum()

        # Add the 'Country' column to the resampled DataFrame
        df_country_resampled['Country'] = country

        dfs_resampled[country] = df_country_resampled

    # Concatenate all the resampled DataFrames into a single DataFrame
    df_resampled = pd.concat(dfs_resampled.values())

    return df_resampled

def combine_dataframes(df_gen, df_load):
    """
    This function combines two DataFrames into one, keeping the same timestamp and adding the 'Load' column to the generation DataFrame.

    Parameters:
    df_gen (pandas.DataFrame): The energy generation DataFrame.
    df_load (pandas.DataFrame): The energy load DataFrame.
    filename (str): The name of the CSV file where the combined DataFrame will be saved.

    Returns:
    df_combined (pandas.DataFrame): The combined DataFrame.
    """

    # Reset the index of both DataFrames
    df_gen = df_gen.reset_index()
    df_load = df_load.reset_index()

    # Combine the two DataFrames into one
    df_combined = pd.merge(df_gen, df_load, on=['StartTime', 'Country'], how='outer')

    # Set 'StartTime' as the index again
    df_combined.set_index('StartTime', inplace=True)

    # Define the green energy columns
    green_energy_columns = ['Biomass', 'Geothermal', 'Hydro Pumped Storage', 
                        'Hydro Run-of-river and poundage', 'Hydro Water Reservoir', 
                        'Marine', 'Other renewable', 'Solar', 'Wind Offshore', 'Wind Onshore']

    # Create a new column 'GE' which is the sum of all the green energy columns
    df_combined['GE'] = df_combined[green_energy_columns].sum(axis=1)

    # Define the column order
    column_order = ['Country', 'Load', 'GE'] + [col for col in df_combined.columns if col not in ['Country', 'Load', 'GE']]

    # Rearrange the columns
    df_combined = df_combined[column_order]

    return df_combined

def melt_reshape_df(df):
    """
    This function reshapes the input DataFrame and returns a new DataFrame with 'StartTime' as the index and columns for 'GE' and 'Load' for each country.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    result (pandas.DataFrame): The reshaped DataFrame.
    """

    # Reset the index to make 'StartTime' a column
    df = df.reset_index()

    # Melt the DataFrame to have 'Country', 'variable' (Load or GE), and 'value' as columns
    df_melt = df.melt(id_vars=['StartTime', 'Country'], value_vars=['Load', 'GE'])

    # Create a new column 'variable_country' that is the concatenation of 'variable' and 'Country'
    df_melt['variable_country'] = df_melt['variable'] + '_' + df_melt['Country']

    # Pivot the DataFrame to get columns for 'Load' and 'GE' for each country
    result = df_melt.pivot(index='StartTime', columns='variable_country', values='value')

    return result
