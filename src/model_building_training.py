import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pmdarima as pm
from tqdm import tqdm
from joblib import dump, load
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools


CC = ['DE', 'DK', 'HU', 'IT', 'NE', 'PO', 'SE', 'SP', 'UK']

def Load_Data(data_path:str='data/Processed Data/EcoForecast Dataset.csv'):
    ecof_df = pd.read_csv(data_path)
    ecof_df = ecof_df.set_index('StartTime')
    print('Data Loaded Succesfully')
    return ecof_df


def fillna_with_window_mean(df, window_size=3):
    """
    Fills missing values in a DataFrame with the mean of the previous and next observed values within a window.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    window_size (int): The size of the window to consider for calculating the mean.

    Returns:
    df_filled (pandas.DataFrame): The DataFrame with missing values filled.
    """

    # Initialize a new DataFrame to store the filled data
    df_filled = df.copy()

    # Convert the index to datetime
    df_filled.index = pd.to_datetime(df_filled.index)

    # For each column in the DataFrame
    for col in df.columns:
        # Get the column data
        data = df_filled[col]

        # Find the indices of missing values
        missing_indices = data.index[data.isna()]

        # For each missing value
        for idx in missing_indices:
            # Find the window of observed values around the missing value
            start_idx = idx - pd.Timedelta(hours=window_size)
            end_idx = idx + pd.Timedelta(hours=window_size)

            prev_values = data.loc[:start_idx].dropna().tail(window_size)
            next_values = data.loc[end_idx:].dropna().head(window_size)

            # Calculate the mean of the observed values in the window
            mean_value = pd.concat([prev_values, next_values]).mean()

            # Fill the missing value with the mean value
            df_filled.loc[idx, col] = mean_value

            # Round Values
            df_filled = df_filled.round(2)

    print('Fill Nan Values by Window Succesfully')

    return df_filled


def split_train_test(df, train_ratio=0.8):
    """
    Splits a DataFrame into a training set and a test set.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    train_ratio (float): The proportion of the data to include in the training set.

    Returns:
    train (pandas.DataFrame): The training set.
    test (pandas.DataFrame): The test set.
    """

    # Calculate the index at which to split the DataFrame
    split_idx = int(len(df) * train_ratio)

    # Split the DataFrame
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    print('Train Test Split 80/20 Succesfully')
    return train, test

def save_df_as_csv(df:pd.DataFrame, filename:str, path:str = 'data/Preprocesed Data/'):
    """
    Saves a DataFrame as a CSV file.

    Parameters:
    df (pandas.DataFrame): The DataFrame to save.
    filename (str): The name of the CSV file.
    """
    print(f'Succesfull Saved: {filename}')
    df.to_csv(filename)

def Get_Surplus_DF(df):
    surplus = pd.DataFrame(index=df.index)
    for country in CC:
        surplus['Surplus_' + country] = np.abs(df['GE_' + country] - df['Load_' + country])
    print('Getting Surplus')
    print(surplus)
    return surplus


def scale_data(df):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    print('Data Succesfully Scaled')
    return df_scaled, scaler

def descale_data(df_scaled, scaler):
    df = pd.DataFrame(scaler.inverse_transform(df_scaled), columns=df_scaled.columns, index=df_scaled.index)
    print('Data Succesfully Descaled')
    return df

def modeling(train_surplus):
    print('Starting Models Building & Fine Tuning\nData Lenght:')
    print(len(train_surplus))
    
    # Define the parameter grid
    param_grid = {
    'changepoint_prior_scale': [0.1, 0.2],
    'seasonality_prior_scale': [1.0, 5.0],
    }

    print(f'The Prophet Forecasting Model is Fine Tuning with Several Hyperparameters.\nIt may cost high time and computation processing\nHyperparameters Settings:\n {param_grid}')
    # Create a dictionary to store the Prophet models
    models = {}

    # Ensure that 'StartTime' is the index and is in datetime format
    train_surplus.index = pd.to_datetime(train_surplus.index)

    # Remove the timezone if it exists
    if train_surplus.index.tz is not None:
        train_surplus.index = train_surplus.index.tz_localize(None)

    # Rename the index to 'ds'
    train_surplus.index.name = 'ds'

    # Train a Prophet model for each country
    for country in tqdm(CC):
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each parameter set here

        # Use cross-validation to evaluate all parameters
        for params in tqdm(all_params):
            m = Prophet(**params).fit(train_surplus[['Surplus_' + country]].reset_index().rename(columns={'Surplus_' + country: 'y'}))  # Fit the model with the given parameters
            train_surplus_cv = cross_validation(m, horizon='2 days', parallel="processes")
            train_surplus_p = performance_metrics(train_surplus_cv, rolling_window=1)
            rmses.append(train_surplus_p['rmse'].values[0])

        # Find the best parameters
        best_params = all_params[np.argmin(rmses)]
        
        # Fit the model with the best parameters
        model = Prophet(**best_params)
        model.fit(train_surplus[['Surplus_' + country]].reset_index().rename(columns={'Surplus_' + country: 'y'}))
        
        models[country] = model


def save_models(models, path='models/'):
    for country, model in models.items():
        # Guardar el modelo en un archivo .joblib
        dump(model, f'{path}model_{country}.joblib')
    print(f'Models Saved Succefully')


def load_models(countries, path='models'):
    models = {}
    for country in countries:
        models[country] = load(f'{path}/model_{country}.joblib')
    print(f'Models Loaded Successfully')
    return models



def Print_MAE(test, pred, n_periods):   
    mae = mean_absolute_error(test[:n_periods], pred)

    print(f"MAE: {mae}")


def ModelBuilding_Pipeline():
    print('Starting Models Building')
    df = Load_Data()
    df = fillna_with_window_mean(df)
    train, test = split_train_test(df)
    save_df_as_csv(train, 'data/train.csv')
    save_df_as_csv(test, 'data/test.csv')
    train_surplus = Get_Surplus_DF(train)
    test_surplus = Get_Surplus_DF(test)
    #strain_surplus, scaler1 = scale_data(train_surplus)
    #stest_surplus, scaler2 = scale_data(test_surplus)
    models_dict = modeling(train_surplus)
    save_models(models_dict)

if __name__ == "__main__":
    ModelBuilding_Pipeline()
   