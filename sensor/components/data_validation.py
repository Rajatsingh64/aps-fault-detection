from sensor.entity import artifact_entity, config_entity
from sensor.exception import SensorException
from sensor.logger import logging
from scipy.stats import ks_2samp
from typing import Optional
import os, sys
import pandas as pd
from sensor import utils
import numpy as np
from sensor.config import TARGET_COLUMN


class DataValidation:
    def __init__(self,
                 data_validation_config: config_entity.DataValidationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        """
        Initializes the DataValidation object with configuration and data ingestion artifact.
        Also initializes an empty dictionary to store validation errors.
        """
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
            logging.info("DataValidation initialization successful.")
        except Exception as e:
            raise SensorException(e, sys)

    def drop_missing_values_columns(self, df: pd.DataFrame, report_key_name: str) -> Optional[pd.DataFrame]:
        """
        Drops columns in the dataframe that have missing values above a specified threshold.
        
        Parameters:
            df (pd.DataFrame): The input dataframe.
            report_key_name (str): Key name for reporting which columns were dropped.
        
        Returns:
            pd.DataFrame or None: The dataframe after dropping the columns, or None if no columns remain.
        """
        try:
            threshold = self.data_validation_config.missing_threshold
            # Calculate the ratio of missing values per column
            null_report = df.isna().sum() / df.shape[0]
            logging.info(f"Calculated missing value report with threshold {threshold}.")
            
            # Identify columns to drop where missing values exceed the threshold
            drop_column_names = null_report[null_report > threshold].index
            logging.info(f"Columns to drop (missing values > {threshold}): {list(drop_column_names)}")
            
            # Store dropped columns in the validation error dictionary
            self.validation_error[report_key_name] = list(drop_column_names)
            
            # Drop the columns from the dataframe
            df.drop(list(drop_column_names), axis=1, inplace=True)
            logging.info("Columns dropped successfully based on missing values threshold.")
            
            # If no columns are left, return None
            if len(df.columns) == 0:
                logging.warning("All columns were dropped after applying missing values threshold.")
                return None
            return df
        except Exception as e:
            raise SensorException(e, sys)

    def is_required_columns_exists(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str) -> bool:
        """
        Checks whether all required columns from the base dataframe exist in the current dataframe.
        
        Parameters:
            base_df (pd.DataFrame): The base dataframe with the required columns.
            current_df (pd.DataFrame): The dataframe to be checked.
            report_key_name (str): Key name for reporting missing columns.
        
        Returns:
            bool: True if all required columns are present; False otherwise.
        """
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns
            missing_columns = []
            
            # Check each column from the base dataframe for presence in the current dataframe
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Required column: [{base_column}] is not available in the current dataframe.")
                    missing_columns.append(base_column)
                    
            if missing_columns:
                self.validation_error[report_key_name] = missing_columns
                logging.warning(f"Missing columns detected: {missing_columns}")
                return False
            
            logging.info("All required columns are present in the current dataframe.")
            return True
        except Exception as e:
            raise SensorException(e, sys)

    def data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str):
        """
        Detects data drift between the base and current dataframes using the KS test.
        Updates the validation_error dictionary with the drift report for each column.
        
        Parameters:
            base_df (pd.DataFrame): The base dataframe.
            current_df (pd.DataFrame): The dataframe to compare against the base.
            report_key_name (str): Key name for reporting data drift.
        """
        try:
            drift_report = dict()
            base_columns = base_df.columns
            
            # Compare each column's distribution between base and current dataframe using the KS test
            for base_column in base_columns:
                base_data = base_df[base_column]
                current_data = current_df[base_column]
                logging.info(f"Performing KS test for column: {base_column}. Data types: {base_data.dtype} vs {current_data.dtype}")
                same_distribution = ks_2samp(base_data, current_data)
                
                if same_distribution.pvalue > 0.05:
                    drift_report[base_column] = {
                        "pvalues": float(same_distribution.pvalue),
                        "same_distribution": True
                    }
                    logging.info(f"Column {base_column} - No significant drift detected (p-value: {same_distribution.pvalue}).")
                else:
                    drift_report[base_column] = {
                        "pvalues": float(same_distribution.pvalue),
                        "same_distribution": False
                    }
                    logging.info(f"Column {base_column} - Data drift detected (p-value: {same_distribution.pvalue}).")
            
            self.validation_error[report_key_name] = drift_report
            logging.info("Data drift report generated successfully.")
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_validation(self) -> artifact_entity.DataValidationArtifact:
        """
        Initiates the data validation process by performing the following steps:
            - Reads the base, train, and test dataframes.
            - Replaces 'na' with NaN in the base dataframe.
            - Drops columns with missing values based on a threshold.
            - Converts columns (except the target column) to float type.
            - Checks for the presence of required columns in train and test dataframes.
            - Detects data drift in train and test dataframes.
            - Writes a YAML report of the validation errors.
        
        Returns:
            artifact_entity.DataValidationArtifact: The artifact containing the path to the YAML report.
        """
        try:
            # Read and prepare the base dataframe
            logging.info("Reading base dataframe.")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({"na": np.NAN}, inplace=True)
            logging.info("Base dataframe read successfully and 'na' values replaced.")
            
            logging.info("Dropping columns with excessive missing values from base dataframe.")
            base_df = self.drop_missing_values_columns(df=base_df, report_key_name="missing_values_within_base_dataset")
            if base_df is None:
                raise SensorException("No columns left in base dataframe after dropping missing value columns.", sys)
            
            # Read the train and test dataframes
            logging.info("Reading train dataframe.")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info("Reading test dataframe.")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            # Drop columns with missing values in train and test dataframes
            logging.info("Dropping columns with excessive missing values from train dataframe.")
            train_df = self.drop_missing_values_columns(df=train_df, report_key_name="missing_values_within_train_dataset")
            logging.info("Dropping columns with excessive missing values from test dataframe.")
            test_df = self.drop_missing_values_columns(df=test_df, report_key_name="missing_values_within_test_dataset")
            
            # Convert columns to float type for base, train, and test dataframes (excluding target column)
            exclude_columns = [TARGET_COLUMN]
            logging.info("Converting columns to float type for base, train, and test dataframes (excluding target column).")
            base_df = utils.convert_columns_float(df=base_df, exclude_columns=exclude_columns)
            train_df = utils.convert_columns_float(df=train_df, exclude_columns=exclude_columns)
            test_df = utils.convert_columns_float(df=test_df, exclude_columns=exclude_columns)
            logging.info("Column conversion to float completed successfully.")
            
            # Check if all required columns exist in train and test dataframes
            logging.info("Validating required columns in train dataframe.")
            train_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=train_df, report_key_name="missing_columns_within_train_dataset")
            logging.info("Validating required columns in test dataframe.")
            test_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=test_df, report_key_name="missing_columns_within_test_dataset")
            
            # Detect data drift if required columns are present
            if train_df_columns_status:
                logging.info("All required columns present in train dataframe. Detecting data drift for train dataset.")
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="data_drift_within_train_dataset")
            else:
                logging.warning("Missing required columns in train dataframe. Skipping data drift detection for train dataset.")
                
            if test_df_columns_status:
                logging.info("All required columns present in test dataframe. Detecting data drift for test dataset.")
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name="data_drift_within_test_dataset")
            else:
                logging.warning("Missing required columns in test dataframe. Skipping data drift detection for test dataset.")
            
            # Write the validation error report to a YAML file
            logging.info("Writing validation error report to YAML file.")
            utils.write_yaml_file(
                file_path=self.data_validation_config.report_file_path,
                data=self.validation_error
            )
            logging.info("Validation error report written successfully.")
            
            # Create the Data Validation Artifact and return it
            data_validation_artifact = artifact_entity.DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path,
            )
            logging.info(f"Data validation artifact created successfully: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise SensorException(e, sys)
