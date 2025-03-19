import json
from textwrap import dedent
import pendulum
import os
from airflow import DAG
from airflow.operators.python import PythonOperator

# ============================================================
# DAG: sensor_training
# Description: This DAG runs the sensor training pipeline and then syncs
#              the generated artifacts and saved models to an S3 bucket.
#              It is scheduled to run weekly.
# ============================================================
with DAG(
    'sensor_training',
    default_args={'retries': 2},
    description='Sensor Fault Detection',
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2025, 3, 13, tz="UTC"),
    catchup=False,
    tags=['example'],
) as dag:

    # ------------------------------------------------------------
    # Task: training
    # Description: Initiates the sensor training pipeline.
    # ------------------------------------------------------------
    def training(**kwargs):
        from sensor.pipeline.training_pipeline import start_training_pipeline
        # Start the training pipeline
        start_training_pipeline()

    # ------------------------------------------------------------
    # Task: sync_artifact_to_s3_bucket
    # Description: Syncs the local 'artifact' and 'saved_models' directories 
    #              to the corresponding folders in an S3 bucket.
    # ------------------------------------------------------------
    def sync_artifact_to_s3_bucket(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        # Sync the artifacts directory to the S3 bucket
        os.system(f"aws s3 sync /app/artifact s3://{bucket_name}/artifacts")
        # Sync the saved models directory to the S3 bucket
        os.system(f"aws s3 sync /app/saved_models s3://{bucket_name}/saved_models")

    # Create the training pipeline task
    training_pipeline = PythonOperator(
        task_id="train_pipeline",
        python_callable=training
    )

    # Create the task to sync data to S3
    sync_data_to_s3 = PythonOperator(
        task_id="sync_data_to_s3",
        python_callable=sync_artifact_to_s3_bucket
    )

    # Define task dependencies: first run training, then sync data to S3
    training_pipeline >> sync_data_to_s3
