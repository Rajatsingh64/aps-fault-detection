import os
from sensor.entity.config_entity import (
    TRANSFORMER_OBJECT_FILE_NAME,
    MODEL_FILE_NAME,
    TARGET_ENCODER_OBJECT_FILE_NAME,
)
from glob import glob
from typing import Optional

# Class for resolving model file paths based on versioning
class ModelResolver:
    def __init__(
        self,
        model_registry: str = "saved_models",
        transformer_dir_name="transformer",
        target_encoder_dir_name="target_encoder",
        model_dir_name="model",
    ):
        # Initialize directories and ensure the model registry exists
        self.model_registry = model_registry
        os.makedirs(self.model_registry, exist_ok=True)
        self.transformer_dir_name = transformer_dir_name
        self.target_encoder_dir_name = target_encoder_dir_name
        self.model_dir_name = model_dir_name

    def get_latest_dir_path(self) -> Optional[str]:
        # Get the latest version directory (highest numeric folder) from the model registry
        try:
            dir_names = os.listdir(self.model_registry)
            if len(dir_names) == 0:
                return None
            dir_names = list(map(int, dir_names))
            latest_dir_name = max(dir_names)
            return os.path.join(self.model_registry, f"{latest_dir_name}")
        except Exception as e:
            raise e

    def get_latest_model_path(self):
        # Return the file path for the latest model
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("Model is not available")
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_transformer_path(self):
        # Return the file path for the latest transformer object
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("Transformer is not available")
            return os.path.join(latest_dir, self.transformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_target_encoder_path(self):
        # Return the file path for the latest target encoder object
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("Target encoder is not available")
            return os.path.join(latest_dir, self.target_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_save_dir_path(self) -> str:
        # Determine the next directory path for saving a new model version
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                return os.path.join(self.model_registry, "0")
            latest_dir_num = int(os.path.basename(latest_dir))
            return os.path.join(self.model_registry, f"{latest_dir_num + 1}")
        except Exception as e:
            raise e

    def get_latest_save_model_path(self):
        # Return the file path to save a new model
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_save_transformer_path(self):
        # Return the file path to save a new transformer object
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.transformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e

    def get_latest_save_target_encoder_path(self):
        # Return the file path to save a new target encoder object
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.target_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e

# Class for making predictions using the resolved model paths
class Predictor:
    def __init__(self, model_resolver: ModelResolver):
        # Initialize Predictor with a ModelResolver instance
        self.model_resolver = model_resolver
