from sensor.predictor import ModelResolver
from sensor.entity import config_entity, artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils import load_object
from sklearn.metrics import f1_score
import pandas as pd
import sys, os
from sensor.config import TARGET_COLUMN
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CLASS: ModelEvaluation
# Description: Evaluates the performance of the current trained model
#              against the previously saved model.
# ============================================================
class ModelEvaluation:

    def __init__(self,
                 model_eval_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainerArtifact):
        """
        Initializes the ModelEvaluation class with configuration and artifacts.
        Also creates a ModelResolver instance for fetching saved model artifacts.
        """
        try:
            logging.info(f"{'>>'*20} Model Evaluation {'<<'*20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
            logging.info("ModelEvaluation initialization successful.")
        except Exception as e:
            raise SensorException(e, sys)

    # ============================================================
    # METHOD: initiate_model_evaluation
    # Description: Compares the performance of the current trained model 
    #              against the previously saved model and returns an 
    #              evaluation artifact.
    # ============================================================
    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info("Comparing current trained model with previously saved model.")

            # Check if a previously saved model exists
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path is None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                    is_model_accepted=True,
                    improved_accuracy=None
                )
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            # ============================================================
            # Step 1: Load previous trained objects (Transformer, Model, Target Encoder)
            # ============================================================
            logging.info("Finding location of transformer model and target encoder from saved models.")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            logging.info("Loading previous trained transformer, model and target encoder objects.")
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)

            # ============================================================
            # Step 2: Load current trained objects (Transformer, Model, Target Encoder)
            # ============================================================
            logging.info("Loading current trained transformer, model and target encoder objects.")
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            # ============================================================
            # Step 3: Evaluate previous trained model on test data
            # ============================================================
            logging.info("Evaluating previous trained model on test data.")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df[TARGET_COLUMN]
            y_true = target_encoder.transform(target_df)

            # Extract features and transform using previous transformer
            input_feature_name = list(transformer.feature_names_in_)
            input_arr = transformer.transform(test_df[input_feature_name])
            y_pred = model.predict(input_arr)
            logging.info(f"Prediction using previous model: {target_encoder.inverse_transform(y_pred[:5])}")
            previous_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"F1 Score using previous trained model: {previous_model_score}")

            # ============================================================
            # Step 4: Evaluate current trained model on test data
            # ============================================================
            logging.info("Evaluating current trained model on test data.")
            input_feature_name = list(current_transformer.feature_names_in_)
            input_arr = current_transformer.transform(test_df[input_feature_name])
            y_pred = current_model.predict(input_arr)
            y_true = current_target_encoder.transform(target_df)
            logging.info(f"Prediction using current model: {current_target_encoder.inverse_transform(y_pred[:5])}")
            current_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"F1 Score using current trained model: {current_model_score}")

            # ============================================================
            # Step 5: Compare model performances
            # ============================================================
            if current_model_score <= previous_model_score:
                logging.info("Current trained model is not better than the previous model.")
                raise Exception("Current trained model is not better than the previous model")

            # Calculate improvement in accuracy
            improved_accuracy = current_model_score - previous_model_score
            logging.info(f"Improved accuracy: {improved_accuracy}")

            # Prepare and return the model evaluation artifact
            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_accuracy=improved_accuracy
            )
            logging.info(f"Model evaluation artifact: {model_eval_artifact}")
            return model_eval_artifact

        except Exception as e:
            raise SensorException(e, sys)
