import os
import sys
import pandas as pd
import yaml
import numpy as np
import dill
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.config import mongo_client


def get_coll_as_df(database_name: str, collection_name: str) -> pd.DataFrame:
    """
    Reads a MongoDB collection and returns its contents as a Pandas DataFrame.

    Parameters:
        database_name (str): Name of the MongoDB database.
        collection_name (str): Name of the collection within the database.

    Returns:
        pd.DataFrame: DataFrame containing the collection's data.
    
    Process:
        - Logs the start of the data reading process.
        - Converts the MongoDB collection to a list of documents and then to a DataFrame.
        - Logs the discovered columns.
        - Removes the '_id' column if present and logs the action.
        - Logs the final shape of the DataFrame before returning it.
    """
    try:
        logging.info(
            f"Reading data from database: {database_name} and collection: {collection_name}"
        )
        # Convert MongoDB collection data to a DataFrame
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")

        # Drop the '_id' column if it exists in the DataFrame
        if "_id" in df.columns:
            logging.info("Dropping column: _id")
            df = df.drop("_id", axis=1)

        logging.info(f"Row and columns in df: {df.shape}")
        return df

    except Exception as e:
        raise SensorException(e, sys)


def write_yaml_file(file_path: str, data: dict):
    """
    Writes a dictionary to a YAML file.

    Parameters:
        file_path (str): The path where the YAML file will be saved.
        data (dict): The dictionary data to write into the YAML file.

    Process:
        - Ensures that the directory for the file exists.
        - Writes the dictionary data to the specified YAML file.
    """
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)

        with open(file_path, "w") as file_writer:
            yaml.dump(data, file_writer)

    except Exception as e:
        raise SensorException(e, sys)


def convert_columns_float(df: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
    """
    Converts all columns in a DataFrame to float, except for the specified columns.

    Parameters:
        df (pd.DataFrame): The DataFrame whose columns need to be converted.
        exclude_columns (list): List of column names to exclude from conversion.

    Returns:
        pd.DataFrame: The DataFrame with converted columns.
    
    Process:
        - Iterates over each column and converts it to float if not in the exclude list.
    """
    try:
        for column in df.columns:
            if column not in exclude_columns:
                df[column] = df[column].astype('float')
        return df

    except Exception as e:
        raise e


def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to a file using dill for serialization.

    Parameters:
        file_path (str): The file path where the object will be saved.
        obj (object): The Python object to be serialized and saved.

    Process:
        - Logs entry into the method.
        - Ensures the directory exists.
        - Serializes and saves the object to the file.
        - Logs exit from the method.
    """
    try:
        logging.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise SensorException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Loads and returns a Python object from a file using dill.

    Parameters:
        file_path (str): The file path from where the object will be loaded.

    Returns:
        object: The deserialized Python object.

    Process:
        - Checks if the file exists.
        - Opens the file and loads the object.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")

        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise SensorException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Saves a NumPy array to a file.

    Parameters:
        file_path (str): The location where the NumPy array will be saved.
        array (np.array): The NumPy array to save.

    Process:
        - Ensures the directory for the file exists.
        - Saves the array to the file in binary format.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise SensorException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Loads a NumPy array from a file.

    Parameters:
        file_path (str): The location of the file containing the NumPy array.

    Returns:
        np.array: The loaded NumPy array.

    Process:
        - Opens the file in binary read mode.
        - Loads and returns the NumPy array.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise SensorException(e, sys) from e
