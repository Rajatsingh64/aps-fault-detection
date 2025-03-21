# Importing necessary modules
from asyncio import tasks  # For asynchronous tasks (though not used directly here)
import json
from textwrap import dedent  # For cleaner multi-line strings (not used here)
import pendulum  # For timezone-aware datetime objects
import os  
from airflow import DAG  # Airflow DAG class
from airflow.operators.python import PythonOperator  # For executing Python callables as tasks

# Define the DAG
with DAG(
    'sensor_prediction',
    default_args={'retries': 2},  # Retry each task up to 2 times if it fails
    description='Sensor Fault Detection',
    schedule_interval="@weekly",  # DAG will run weekly
    start_date=pendulum.datetime(2025, 3, 13, tz="UTC"),  # Start date of the DAG
    catchup=False,  # Do not run missed DAG runs between start_date and now
    tags=['example'],  # Tagging the DAG
) as dag:

    # Task 1: Download files from S3 bucket to local directory
    def download_files(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")  # Get S3 bucket name from environment variable
        input_dir = "/app/input_files"  # Local directory to store input files
        
        # Create input directory if it doesn't exist
        os.makedirs(input_dir, exist_ok=True)
        
        # Sync files from S3 bucket to local input directory
        os.system(f"aws s3 sync s3://{bucket_name}/input_files /app/input_files")

    # Task 2: Run batch prediction on each input file
    def batch_prediction(**kwargs):
        from sensor.pipeline.batch_prediction import start_batch_prediction  # Import batch prediction function
        input_dir = "/app/input_files"  # Local input directory
        
        # Iterate through each file in the input directory
        for file_name in os.listdir(input_dir):
            # Run batch prediction for each file
            start_batch_prediction(input_file_path=os.path.join(input_dir, file_name))
    
    # Task 3: Sync the prediction results back to S3 bucket
    def sync_prediction_dir_to_s3_bucket(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")  # Get S3 bucket name from environment variable
        
        # Upload local prediction folder to prediction_files folder in S3 bucket
        os.system(f"aws s3 sync /app/prediction s3://{bucket_name}/prediction_files")
    
    # Define the PythonOperator for downloading input files
    download_input_files = PythonOperator(
        task_id="download_file",  # Task ID
        python_callable=download_files  # Python function to execute
    )

    # Define the PythonOperator for generating predictions
    generate_prediction_files = PythonOperator(
        task_id="prediction",  # Task ID
        python_callable=batch_prediction  # Python function to execute
    )

    # Define the PythonOperator for uploading prediction files to S3
    upload_prediction_files = PythonOperator(
        task_id="upload_prediction_files",  # Task ID
        python_callable=sync_prediction_dir_to_s3_bucket  # Python function to execute
    )

    # Set task dependencies: download -> predict -> upload
    download_input_files >> generate_prediction_files >> upload_prediction_files
