from sensor.entity import artifact_entity, config_entity
from sensor.exception import SensorException
from sensor.logger import logging
from typing import Optional
import os, sys
from sklearn.pipeline import Pipeline
import pandas as pd
from sensor import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sensor.config import TARGET_COLUMN

class DataTransformation:
    def __init__(self, 
                 data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        """
        Initialize the DataTransformation class with configuration and 
        data ingestion artifact.
        """
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            logging.info("Data Transformation initialization successful.")
        except Exception as e:
            raise SensorException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        """
        Create and return a transformation pipeline that applies:
        1. SimpleImputer: To fill missing values with a constant value (0).
        2. RobustScaler: To scale features using statistics that are robust to outliers.
        """
        try:
            # Create an imputer to fill missing values with 0
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            # Create a robust scaler for scaling features
            robust_scaler = RobustScaler()
            # Build the pipeline with imputer and scaler
            pipeline = Pipeline(steps=[
                ('Imputer', simple_imputer),
                ('RobustScaler', robust_scaler)
            ])
            logging.info("Data transformer object created successfully.")
            return pipeline
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        """
        Execute the data transformation process:
        1. Read train and test data.
        2. Separate input features and target variable.
        3. Encode the target variable.
        4. Fit and apply the transformation pipeline to input features.
        5. Apply SMOTETomek to balance the datasets.
        6. Save the transformed arrays, pipeline, and label encoder.
        7. Create and return the DataTransformationArtifact.
        """
        try:
            # Read the training and testing files
            logging.info("Reading training and testing files.")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info("Training and testing files read successfully.")

            # Separate input features by dropping the target column
            logging.info("Selecting input features by dropping target column.")
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            # Select the target feature
            logging.info("Selecting target feature for train and test datasets.")
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Encode the target column using LabelEncoder
            logging.info("Encoding target column using LabelEncoder.")
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)
            logging.info("Target column encoded successfully.")

            # Get the data transformation pipeline and fit it on training input features
            logging.info("Creating and fitting data transformation pipeline on training data.")
            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            # Transform the input features for both train and test datasets
            logging.info("Transforming input features for training and testing datasets.")
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)
            logging.info("Input features transformed successfully.")

            # Apply SMOTETomek for resampling to balance the target variable in training data
            smt = SMOTETomek(random_state=42)
            logging.info(f"Before resampling (train set): Input shape {input_feature_train_arr.shape}, "
                         f"Target shape {target_feature_train_arr.shape}")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling (train set): Input shape {input_feature_train_arr.shape}, "
                         f"Target shape {target_feature_train_arr.shape}")

            # Apply SMOTETomek for resampling on testing data as well
            logging.info(f"Before resampling (test set): Input shape {input_feature_test_arr.shape}, "
                         f"Target shape {target_feature_test_arr.shape}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"After resampling (test set): Input shape {input_feature_test_arr.shape}, "
                         f"Target shape {target_feature_test_arr.shape}")

            # Combine the resampled input features and target variables for train and test sets
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            # Save the transformed training and testing arrays as numpy files
            logging.info("Saving transformed training and testing arrays.")
            utils.save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_path,
                array=train_arr
            )
            utils.save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_path,
                array=test_arr
            )
            logging.info("Transformed arrays saved successfully.")

            # Save the transformation pipeline and label encoder objects
            logging.info("Saving transformation pipeline and target encoder objects.")
            utils.save_object(
                file_path=self.data_transformation_config.transform_object_path,
                obj=transformation_pipeline
            )
            utils.save_object(
                file_path=self.data_transformation_config.target_encoder_path,
                obj=label_encoder
            )
            logging.info("Transformation pipeline and target encoder saved successfully.")

            # Create the data transformation artifact containing file paths for the saved objects
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path,
                target_encoder_path=self.data_transformation_config.target_encoder_path
            )
            logging.info(f"Data transformation artifact created successfully: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys)
