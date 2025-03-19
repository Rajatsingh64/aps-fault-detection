from sensor import utils
from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
import os, sys
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        try:
            # Export collection data as a pandas DataFrame
            logging.info("Exporting collection data as pandas dataframe")
            df: pd.DataFrame = utils.get_coll_as_df(
                database_name=self.data_ingestion_config.database_name, 
                collection_name=self.data_ingestion_config.collection_name
            )
            logging.info("Data exported from collection successfully.")

            # Replace "na" values with NaN
            logging.info("Replacing 'na' with NaN in dataframe")
            df.replace(to_replace="na", value=np.NAN, inplace=True)
            logging.info("Missing values replaced successfully.")

            # Save data in feature store
            logging.info("Creating feature store folder if not available")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            logging.info("Feature store folder created successfully.")

            logging.info("Saving dataframe to feature store folder")
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path, index=False, header=True)
            logging.info("Data saved to feature store successfully.")

            # Split dataset into train and test sets
            logging.info("Splitting dataset into train and test sets")
            train_df, test_df = train_test_split(
                df, test_size=self.data_ingestion_config.test_size, random_state=42
            )
            logging.info("Dataset split into train and test sets successfully.")

            # Create dataset directory folder if not available
            logging.info("Creating dataset directory folder if not available")
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)
            logging.info("Dataset directory folder created successfully.")

            # Save train and test datasets
            logging.info("Saving train and test datasets to dataset directory")
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path, index=False, header=True)
            logging.info("Train and test datasets saved successfully.")

            # Prepare the Data Ingestion artifact with file paths
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path, 
                test_file_path=self.data_ingestion_config.test_file_path
            )
            logging.info(f"Data ingestion artifact created successfully: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise SensorException(e, sys)
