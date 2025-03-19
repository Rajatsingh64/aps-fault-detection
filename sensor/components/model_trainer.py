from sensor.entity import artifact_entity, config_entity
from sensor.exception import SensorException
from sensor.logger import logging
from typing import Optional
import os, sys
from xgboost import XGBClassifier
from sensor import utils
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self,
                 model_trainer_config: config_entity.ModelTrainerConfig,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact):
        """
        Initializes the ModelTrainer with configuration settings and 
        data transformation artifacts.
        """
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            logging.info("ModelTrainer initialization successful.")
        except Exception as e:
            raise SensorException(e, sys)

    def fine_tune(self):
        """
        Placeholder for fine tuning using Grid Search CV.
        """
        try:
            # TODO: Implement fine tuning using GridSearchCV if needed.
            pass
        except Exception as e:
            raise SensorException(e, sys)

    def train_model(self, x, y):
        """
        Trains an XGBoost classifier using the provided features and target.
        
        Parameters:
            x (np.array): Input feature array.
            y (np.array): Target variable array.
        
        Returns:
            model: Trained XGBoost model.
        """
        try:
            logging.info("Starting model training using XGBClassifier.")
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x, y)
            logging.info("Model training completed successfully.")
            return xgb_clf
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_trainer(self) -> artifact_entity.ModelTrainerArtifact:
        """
        Initiates the model training process:
            1. Loads the transformed training and testing arrays.
            2. Splits them into input features and target variables.
            3. Trains an XGBoost classifier.
            4. Calculates F1 scores for train and test sets.
            5. Checks for underfitting and overfitting based on configured thresholds.
            6. Saves the trained model.
            7. Returns a ModelTrainerArtifact containing model path and scores.
        
        Returns:
            artifact_entity.ModelTrainerArtifact: Artifact with model path and performance scores.
        """
        try:
            logging.info("Loading transformed training and testing arrays.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)  # Note: Correct attribute name should be model_trainer_config
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)
            logging.info("Arrays loaded successfully.")

            # Split the arrays into input features and target variable
            logging.info("Splitting input features and target variable from training and testing arrays.")
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            logging.info(f"Training set shape: {x_train.shape}, Testing set shape: {x_test.shape}")

            # Train the model using the training dataset
            logging.info("Training the model.")
            model = self.train_model(x=x_train, y=y_train)

            # Calculate F1 score on training data
            logging.info("Calculating F1 score on training data.")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train, y_pred=yhat_train)
            logging.info(f"Training F1 score: {f1_train_score}")

            # Calculate F1 score on testing data
            logging.info("Calculating F1 score on testing data.")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test)
            logging.info(f"Testing F1 score: {f1_test_score}")

            # Check if the model meets the expected performance criteria
            logging.info("Checking if the model meets the expected performance criteria.")
            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(
                    f"Model did not meet the expected accuracy: {self.model_trainer_config.expected_score}. "
                    f"Actual test score: {f1_test_score}"
                )

            # Check for overfitting: difference between train and test F1 scores
            diff = abs(f1_train_score - f1_test_score)
            logging.info(f"Difference between training and testing F1 scores: {diff}")
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(
                    f"Difference between training and testing scores ({diff}) exceeds the overfitting threshold "
                    f"{self.model_trainer_config.overfitting_threshold}"
                )

            # Save the trained model to the specified file path
            logging.info("Saving the trained model.")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)
            logging.info("Model saved successfully.")

            # Prepare the model trainer artifact containing the model path and performance scores
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path=self.model_trainer_config.model_path,
                f1_train_score=f1_train_score,
                f1_test_score=f1_test_score
            )
            logging.info(f"Model Trainer Artifact created successfully: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e, sys)
