from dataclasses import dataclass

# ============================================================
# CLASS: DataIngestionArtifact
# Description: Holds file paths related to the data ingestion process,
#              including feature store, training, and testing datasets.
# ============================================================
@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str  # Path to the feature store CSV file
    train_file_path: str          # Path to the training dataset CSV file
    test_file_path: str           # Path to the testing dataset CSV file

# ============================================================
# CLASS: DataValidationArtifact
# Description: Holds file path for the data validation report.
# ============================================================
@dataclass
class DataValidationArtifact:
    report_file_path: str         # Path to the YAML report file generated during data validation

# ============================================================
# CLASS: DataTransformationArtifact
# Description: Contains file paths for the transformed data and related objects,
#              including transformer and target encoder.
# ============================================================
@dataclass
class DataTransformationArtifact:
    transform_object_path: str    # Path to the saved transformer object (e.g., scaler, encoder)
    transformed_train_path: str   # Path to the transformed training dataset file (e.g., NPZ format)
    transformed_test_path: str    # Path to the transformed testing dataset file (e.g., NPZ format)
    target_encoder_path: str      # Path to the saved target encoder object

# ============================================================
# CLASS: ModelTrainerArtifact
# Description: Contains artifacts generated during model training,
#              including the model file and evaluation scores.
# ============================================================
@dataclass
class ModelTrainerArtifact:
    model_path: str               # Path where the trained model is saved
    f1_train_score: float         # F1 score on the training dataset
    f1_test_score: float          # F1 score on the testing dataset

# ============================================================
# CLASS: ModelEvaluationArtifact
# Description: Holds the outcome of the model evaluation step,
#              indicating if the model is accepted and the improvement in accuracy.
# ============================================================
@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool       # Flag indicating if the model meets acceptance criteria
    improved_accuracy: float      # The improvement in accuracy compared to the previous model

# ============================================================
# CLASS: ModelPusherArtifact
# Description: Contains paths related to the model pushing process,
#              including the directory for the pushed model and the saved model directory.
# ============================================================
@dataclass
class ModelPusherArtifact:
    pusher_model_dir: str         # Directory where the model is pushed for production
    saved_model_dir: str          # Directory where the model is saved as the production model
