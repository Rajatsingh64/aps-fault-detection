from sensor.logger import logging
from sensor.exception import SensorException
import sys, os
from sensor.utils import get_coll_as_df
from sensor.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher
import warnings
warnings.filterwarnings("ignore")


def start_training_pipeline():
    try:
        print(f"{'>'*10} Starting Training Pipeline {'<'*10}")
        logging.info(f"{'>'*10} Starting Training Pipeline {'<'*10}")

        # Initialize Training Pipeline Config
        training_pipe_config = TrainingPipelineConfig()

        # -------------------- Data Ingestion --------------------
        print(f"{'>'*10} Starting Data Ingestion {'<'*10}")
        logging.info(f"{'>'*10} Starting Data Ingestion {'<'*10}")

        data_ingestion_config = DataIngestionConfig(training_pipe_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        print(f"{'>'*10} Data Ingestion Completed {'<'*10}")
        logging.info(f"{'>'*10} Data Ingestion Completed {'<'*10}")

        # -------------------- Data Validation --------------------
        print(f"{'>'*10} Starting Data Validation {'<'*10}")
        logging.info(f"{'>'*10} Starting Data Validation {'<'*10}")

        data_validation_config = DataValidationConfig(training_pipe_config)
        data_validation = DataValidation(
            data_validation_config=data_validation_config,
            data_ingestion_artifact=data_ingestion_artifact
        )
        data_validation_artifact = data_validation.initiate_data_validation()

        print(f"{'>'*10} Data Validation Completed {'<'*10}")
        logging.info(f"{'>'*10} Data Validation Completed {'<'*10}")

        # -------------------- Data Transformation --------------------
        print(f"{'>'*10} Starting Data Transformation {'<'*10}")
        logging.info(f"{'>'*10} Starting Data Transformation {'<'*10}")

        data_transformation_config = DataTransformationConfig(training_pipe_config)
        data_transformation = DataTransformation(
            data_transformation_config=data_transformation_config,
            data_ingestion_artifact=data_ingestion_artifact
        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()

        print(f"{'>'*10} Data Transformation Completed {'<'*10}")
        logging.info(f"{'>'*10} Data Transformation Completed {'<'*10}")

        # -------------------- Model Training --------------------
        print(f"{'>'*10} Starting Model Training {'<'*10}")
        logging.info(f"{'>'*10} Starting Model Training {'<'*10}")

        model_trainer_config = ModelTrainerConfig(training_pipe_config)
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        print(f"{'>'*10} Model Training Completed {'<'*10}")
        logging.info(f"{'>'*10} Model Training Completed {'<'*10}")

        # -------------------- Model Evaluation --------------------
        print(f"{'>'*10}  Starting Model Evaluation {'<'*10}")
        logging.info(f"{'>'*10}  Starting Model Evaluation {'<'*10}")

        model_eva_config = ModelEvaluationConfig(training_pipeline_config=training_pipe_config)
        model_evaluation = ModelEvaluation(
            model_eval_config=model_eva_config,
            data_ingestion_artifact=data_ingestion_artifact,
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_artifact=model_trainer_artifact
        )
        model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

        print(f"{'>'*10} Model Evaluation Completed {'<'*10}")
        logging.info(f"{'>'*10} Model Evaluation Completed {'<'*10}")

        # -------------------- Model Pusher --------------------
        print(f"{'>'*10} Starting Model Pusher {'<'*10}")
        logging.info(f"{'>'*10}  Starting Model Pusher {'<'*10}")

        model_pusher_config = ModelPusherConfig(training_pipeline_config=training_pipe_config)
        model_pusher = ModelPusher(
            data_transformation_artifact=data_transformation_artifact,
            model_pusher_config=model_pusher_config,
            model_trainer_artifact=model_trainer_artifact
        )
        model_pusher_artifact = model_pusher.initiate_model_pusher()

        print(f"{'>'*10} Model Pusher Completed {'<'*10}")
        logging.info(f"{'>'*10} Model Pusher Completed {'<'*10}")

        print(f"{'>'*10} Training Pipeline Completed Successfully! {'<'*10}")
        logging.info(f"{'>'*10} Training Pipeline Completed Successfully! {'<'*10}")

    except Exception as e:
        logging.error(f" Exception occurred in Training Pipeline: {e}")
        raise SensorException(e, sys)
