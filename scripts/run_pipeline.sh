#!/bin/bash

# You can run this script from the command line using:
# ./run_pipeline.sh <start_date> <end_date> <raw_data_file> <processed_data_file> <model_file> <test_data_file> <predictions_file>
# For example:
# ./run_pipeline.sh 2020-01-01 2020-01-31 data/raw_data.csv data/processed_data.csv models/model.pkl data/test_data.csv predictions/predictions.json

# Get command line arguments
# start_date="01-01-2022"
# end_date="01-01-2023"
# raw_data_file1='data/Processed Data/merged_gen_data.csv'
# raw_data_file2='data/Processed Data/load_gen_data.csv'
# processed_data_file='data/Processed Data/EcoForecast Datasetv1.csv'
# model_file="$5"
# test_data_file="$6"
# predictions_file="$7"

# Run data_ingestion.py
echo "Starting Data Ingestion..."
python src/data_ingestion.py
# Run ETL
echo "Starting ETL..."
python src/etl.py
# Run data_processing.py
echo "Starting Data Processing..."
python src/data_processing.py 
# Run model_training.py
echo "Starting Model Building & Training..."
python src/model_building_training.py 
# Run model_prediction.py
echo "Starting Prediction..."
python src/model_prediction.py 

echo "Pipeline completed."
