import os
import sys
from sensor.exception import SensorException
from sensor.logger import logging
from datetime import datetime

# ============================================================
# CONSTANTS
# ============================================================
FILE_NAME = "sensor.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"

# ============================================================
# CLASS: TrainingPipelineConfig
# Description: Holds configuration for the overall training pipeline,
#              including the artifact directory where all outputs will be stored.
# ============================================================
class TrainingPipelineConfig:
    def __init__(self):
        try:
            # Create an artifact directory with a timestamp for unique versioning
            self.artifact_dir = os.path.join(
                os.getcwd(), "artifact", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}"
            )
        except Exception as e:
            raise SensorException(e, sys)

# ============================================================
# CLASS: DataIngestionConfig
# Description: Configuration settings for the data ingestion stage,
#              including database and file paths for the raw dataset.
# ============================================================
class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.database_name = "aps"
            self.collection_name = "sensor"
            # Directory for storing ingestion-related artifacts
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
            # File path for storing the feature store CSV
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, "feature_store", FILE_NAME)
            # File paths for the training and testing datasets
            self.train_file_path = os.path.join(self.data_ingestion_dir, "dataset", TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir, "dataset", TEST_FILE_NAME)
            # Fraction of the dataset to be used as the test set
            self.test_size = 0.2
        except Exception as e:
            raise SensorException(e, sys)

    # ============================================================
    # METHOD: to_dict
    # Description: Converts the configuration attributes into a dictionary.
    # ============================================================
    def to_dict(self) -> dict:
        try:
            return self.__dict__
        except Exception as e:
            raise SensorException(e, sys)

# ============================================================
# CLASS: DataValidationConfig
# Description: Configuration for data validation stage,
#              including paths for reports and thresholds for missing values.
# ============================================================
class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory for storing data validation artifacts
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
        # File path for the YAML report of data validation
        self.report_file_path = os.path.join(self.data_validation_dir, "report.yaml")
        # Threshold for acceptable missing values in the dataset
        self.missing_threshold: float = 0.2
        # Base file path for the validation dataset
        self.base_file_path = os.path.join("aps_failure_training_set1.csv")

# ============================================================
# CLASS: DataTransformationConfig
# Description: Configuration for transforming the data prior to model training,
#              including paths for saving transformer objects and transformed datasets.
# ============================================================
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory for storing data transformation artifacts
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")
        # File path to save the transformer object (e.g., scaler, encoder)
        self.transform_object_path = os.path.join(self.data_transformation_dir, "transformer", TRANSFORMER_OBJECT_FILE_NAME)
        # File paths for saving the transformed training and test datasets in NPZ format
        self.transformed_train_path = os.path.join(self.data_transformation_dir, "transformed", TRAIN_FILE_NAME.replace("csv", "npz"))
        self.transformed_test_path = os.path.join(self.data_transformation_dir, "transformed", TEST_FILE_NAME.replace("csv", "npz"))
        # File path to save the target encoder object
        self.target_encoder_path = os.path.join(self.data_transformation_dir, "target_encoder", TARGET_ENCODER_OBJECT_FILE_NAME)

# ============================================================
# CLASS: ModelTrainerConfig
# Description: Configuration for model training,
#              including model saving paths, expected performance, and overfitting threshold.
# ============================================================
class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory for storing model trainer artifacts
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
        # File path for saving the trained model
        self.model_path = os.path.join(self.model_trainer_dir, "model", MODEL_FILE_NAME)
        # Expected minimum score for the model performance
        self.expected_score = 0.7
        # Threshold to detect overfitting
        self.overfitting_threshold = 0.1

# ============================================================
# CLASS: ModelEvaluationConfig
# Description: Configuration for evaluating the model,
#              including the threshold for performance change.
# ============================================================
class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Threshold to determine if the model performance change is significant
        self.change_threshold = 0.01

# ============================================================
# CLASS: ModelPusherConfig
# Description: Configuration for pushing the model to production,
#              including file paths for saving the pushed model and related objects.
# ============================================================
class ModelPusherConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory for storing model pusher artifacts
        self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir, "model_pusher")
        # Directory where the saved models will be maintained in production
        self.saved_model_dir = os.path.join("saved_models")
        # Directory for the pushed model version
        self.pusher_model_dir = os.path.join(self.model_pusher_dir, "saved_models")
        # File paths for saving the model, transformer, and target encoder in the pusher directory
        self.pusher_model_path = os.path.join(self.pusher_model_dir, MODEL_FILE_NAME)
        self.pusher_transformer_path = os.path.join(self.pusher_model_dir, TRANSFORMER_OBJECT_FILE_NAME)
        self.pusher_target_encoder_path = os.path.join(self.pusher_model_dir, TARGET_ENCODER_OBJECT_FILE_NAME)
