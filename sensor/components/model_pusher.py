from sensor.predictor import ModelResolver
from sensor.entity.config_entity import ModelPusherConfig
from sensor.exception import SensorException
import os, sys
from sensor.utils import load_object, save_object
from sensor.logger import logging
from sensor.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelPusherArtifact

class ModelPusher:
    def __init__(self,
                 model_pusher_config: ModelPusherConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        """
        Initializes the ModelPusher with configuration and artifact information.
        Also initializes a ModelResolver instance for managing the saved model directory.
        """
        try:
            logging.info(f"{'>>'*20} Model Pusher Initialization {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            # Initialize ModelResolver with the directory for saved models
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
            logging.info("ModelPusher initialization successful.")
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Initiates the model pusher process:
          1. Loads the transformer, model, and target encoder objects from the current artifacts.
          2. Saves these objects into the model pusher directory.
          3. Saves the same objects into the saved model directory for future reference.
          4. Returns a ModelPusherArtifact containing the relevant directories.
        """
        try:
            # --------------------------------------------------------------------
            # Step 1: Load current trained objects
            # --------------------------------------------------------------------
            logging.info("Loading transformer, model, and target encoder from artifacts.")
            transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model = load_object(file_path=self.model_trainer_artifact.model_path)
            target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)
            logging.info("Successfully loaded current trained objects.")

            # --------------------------------------------------------------------
            # Step 2: Save objects into the model pusher directory
            # --------------------------------------------------------------------
            logging.info("Saving objects into the model pusher directory.")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
            save_object(file_path=self.model_pusher_config.pusher_target_encoder_path, obj=target_encoder)
            logging.info("Objects saved successfully in the model pusher directory.")

            # --------------------------------------------------------------------
            # Step 3: Save objects into the saved model directory using ModelResolver
            # --------------------------------------------------------------------
            logging.info("Saving objects into the saved model directory.")
            transformer_path = self.model_resolver.get_latest_save_transformer_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            target_encoder_path = self.model_resolver.get_latest_save_target_encoder_path()
            save_object(file_path=transformer_path, obj=transformer)
            save_object(file_path=model_path, obj=model)
            save_object(file_path=target_encoder_path, obj=target_encoder)
            logging.info("Objects saved successfully in the saved model directory.")

            # --------------------------------------------------------------------
            # Step 4: Prepare and return the ModelPusherArtifact
            # --------------------------------------------------------------------
            model_pusher_artifact = ModelPusherArtifact(
                pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                saved_model_dir=self.model_pusher_config.saved_model_dir
            )
            logging.info(f"Model pusher artifact created successfully: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise SensorException(e, sys)
