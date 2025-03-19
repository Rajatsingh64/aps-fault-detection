from sensor.exception import SensorException
from sensor.logger import logging
from sensor.predictor import ModelResolver
import pandas as pd
from sensor.utils import load_object
import os, sys
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PREDICTION_DIR = "prediction"

def start_batch_prediction(input_file_path):
    """
    Runs the batch prediction pipeline:
    1. Creates a prediction directory.
    2. Reads the input CSV file.
    3. Loads the latest transformer to transform the input data.
    4. Loads the latest model to make predictions.
    5. Loads the target encoder to convert numeric predictions to categorical labels.
    6. Saves the prediction results to a new CSV file.
    
    Parameters:
        input_file_path (str): Path to the input CSV file.
        
    Returns:
        str: Path to the CSV file containing predictions.
    """
    try:
        # Create the prediction directory if it doesn't exist.
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        logging.info("Prediction directory created or already exists.")
        
        # Create the model resolver object to fetch the latest model components.
        logging.info("Creating model resolver object.")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info("Model resolver object created successfully.")
        
        # Read the input CSV file into a DataFrame.
        logging.info(f"Reading file: {input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"na": np.NAN}, inplace=True)
        logging.info("Input file read successfully and missing values replaced.")
        
        # Load the transformer to transform the dataset.
        logging.info("Loading transformer to transform dataset.")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
        logging.info("Transformer loaded successfully.")
        
        # Extract input feature names from the transformer and transform the DataFrame.
        input_feature_names = list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_names])
        logging.info("Dataset transformed successfully using the transformer.")
        
        # Load the latest model to make predictions.
        logging.info("Loading model to make prediction.")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        logging.info("Model loaded successfully.")
        prediction = model.predict(input_arr)
        logging.info("Predictions generated successfully.")
        
        # Load the target encoder to convert predictions to categorical labels.
        logging.info("Loading target encoder to convert predictions to categorical labels.")
        target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())
        cat_prediction = target_encoder.inverse_transform(prediction)
        logging.info("Target encoder applied successfully.")
        
        # Add the prediction results to the DataFrame.
        df["prediction"] = prediction
        df["cat_pred"] = cat_prediction
        
        # Create a prediction file name with a timestamp.
        prediction_file_name = os.path.basename(input_file_path).replace(
            ".csv", f"_{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv"
        )
        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name)
        
        # Save the DataFrame with predictions to a CSV file.
        df.to_csv(prediction_file_path, index=False, header=True)
        logging.info(f"Prediction file saved successfully at: {prediction_file_path}")
        
        # Log the successful completion of the batch prediction process.
        logging.info("Batch prediction process completed successfully")
        return prediction_file_path
        
    except Exception as e:
        raise SensorException(e, sys)
