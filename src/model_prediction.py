import pandas as pd
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from model_building_training import scale_data, descale_data, Get_Surplus_DF, load_models, CC, Print_MAE
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def predict_surplus(surplus, n_periods, models):
    print('Starting Forecast...')
    # Crear un DataFrame para las predicciones
    predictions = pd.DataFrame()
    
    # Hacer predicciones para cada país
    for country in CC:
        predictions['Surplus_' + country] = models[country].predict(n_periods=n_periods)
    
    # Crear el índice de tiempo para las predicciones
    last_date = surplus.index[-1]
    predictions.index = pd.date_range(start=last_date, periods=n_periods+1, closed='right', freq='H')
    print('Forecast Succesfully')
    return predictions

def Print_MAE(test, pred, n_periods):   
    mae = mean_absolute_error(test[:n_periods], pred)

    print(f"MAE: {mae}")


def load_data(file_path, index_col):
    df = pd.read_csv(file_path, index_col=index_col)
    return df

def load_model(CC, model_path):
    models_dict = load_models(CC, model_path)
    return models_dict

def predict_surplus(test_surplus, n_periods, models):
    print('Starting Green Energy Surplus Prediction...')
    # Crear un DataFrame para las predicciones
    predictions = pd.DataFrame()
    #test_surplus.set_index('StartTime')
    
    # Hacer predicciones para cada país
    for country in CC:
        # Crear un DataFrame para las predicciones futuras
        future = models[country].make_future_dataframe(periods=n_periods, freq='H')
        # Realizar la predicción
        forecast = models[country].predict(future)
        # Añadir las predicciones al DataFrame de predicciones
        predictions['Surplus_' + country] = forecast['yhat'].tail(n_periods).values
    
    # Crear el índice de tiempo para las predicciones
    start_date = pd.to_datetime('2022-10-19 23:00:00')
    predictions.index = pd.date_range(start=start_date, periods=n_periods, freq='H')
    #predictions.set_index('StartTime')
    predictions.index = pd.to_datetime(predictions.index)
    # # Drop Time Zone
    if predictions.index.tz is not None:
        predictions.index = predictions.index.tz_localize(None)
    
    print(f'Forecasted Values: \n{predictions}')
    return predictions


def plot_predictions(test, predictions):
    # Diccionario de códigos de países
    country_codes = {
        'SP': 'Spain',
        'UK': 'United Kingdom',
        'DE': 'Germany',
        'DK': 'Denmark',
        'HU': 'Hungary',
        'SE': 'Sweden',
        'IT': 'Italy',
        'PO': 'Poland',
        'NE': 'Netherlands'
    }
    #test.index = pd.to_datetime(test.index)
    #predictions.index = pd.to_datetime(predictions.index)

    # Crear una figura con subplots
    fig, axs = plt.subplots(len(country_codes), 1, figsize=(10, 6*len(country_codes)))
    
    # Para cada país
    for i, (code, country) in enumerate(country_codes.items()):
        # Extraer la serie de tiempo real y predicha
        real = test['Surplus_' + code]
        predicted = predictions['Surplus_' + code]
        
        # Trazar la serie de tiempo real y predicha en el subplot correspondiente
        axs[i].plot(real.index, real, label='Real')
        axs[i].plot(predicted.index, predicted, label='Predicted')
        
        # Añadir título y etiquetas al subplot
        axs[i].set_title('Green Energy Surplus for ' + country)
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Surplus')
        axs[i].legend()
    
    # Ajustar el espacio entre subplots
    plt.tight_layout()
    # Guardar el gráfico como una imagen
    plt.savefig('images/EcoForcast.png')
    # Mostrar la figura
    #plt.show()


def find_max_surplus_country(predictions):
    country_codes_num = {
    'Surplus_SP': 0, # Spain
    'Surplus_UK': 1, # United Kingdom
    'Surplus_DE': 2, # Germany
    'Surplus_DK': 3, # Denmark
    'Surplus_HU': 5, # Hungary
    'Surplus_SE': 4, # Sweden
    'Surplus_IT': 6, # Italy
    'Surplus_PO': 7, # Poland
    'Surplus_NE': 8 # Netherlands
}

    max_surplus_country = predictions.idxmax(axis=1)

    max_surplus_country = max_surplus_country.replace(country_codes_num)

    max_surplus_country.index = range(len(max_surplus_country))

    results = {'target': max_surplus_country.to_dict()}

    with open('predictions/predictions.json', 'w') as f:
        json.dump(results, f)

    return results

def make_predictions(df, n_periods,model):
    predictions = predict_surplus(df, n_periods, model)
    #dscaled_predictions = descale_data(predictions, scaler)
    return predictions

def save_predictions(predictions):
    find_max_surplus_country(predictions)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/test.csv', 
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions.json', 
        help='Path to save the predictions'
    )
    parser.add_argument(
        '--n_periods', 
        type=int, 
        default=442, 
        help='Number of Periods to Predict'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file, n_periods):
    test = load_data(input_file,'StartTime')
    test_surplus = Get_Surplus_DF(test)
    
    print('Real Values: \n')
    print(test_surplus)
    
    model_dict = load_model(CC,model_file)
    #stest_surplus, scaler = scale_data(test_surplus)
    predictions = make_predictions(test_surplus,n_periods,model_dict)
    Print_MAE(test_surplus, predictions,n_periods)
    save_predictions(predictions)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file, args.n_periods)
    input('Press Enter to Continue...')
