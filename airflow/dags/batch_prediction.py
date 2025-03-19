# Standard library imports
import os

# External library imports
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

# Task 1: Download input files from S3 to local directory
def download_files(**kwargs):
    """
    Downloads input files from the configured S3 bucket to the local input directory.

    Steps:
    1. Fetch bucket name from environment variable.
    2. Create local directory '/app/input_files' if it doesn't exist.
    3. Use S3Hook to list all keys/files under 'input_files/' prefix in the S3 bucket.
    4. Download each file to the local input directory.
    """
    try:
        bucket_name = os.getenv("BUCKET_NAME")  # S3 bucket name from env
        input_dir = "/app/input_files"
        os.makedirs(input_dir, exist_ok=True)  # Ensure input directory exists
        
        s3_hook = S3Hook()  # Initialize S3 Hook (uses Airflow connection)
        
        # List all file keys under 'input_files/' prefix
        keys = s3_hook.list_keys(bucket_name=bucket_name, prefix="input_files/")
        if not keys:
            raise Exception(f"No files found in S3 bucket {bucket_name} under prefix 'input_files/'")
        
        # Download each file from S3 to local input directory
        for key in keys:
            local_path = os.path.join(input_dir, os.path.basename(key))
            s3_hook.download_file(key=key, bucket_name=bucket_name, local_path=local_path)
            print(f"Downloaded {key} to {local_path}")
    
    except Exception as e:
        print(f"Error downloading files: {e}")
        raise e

# Task 2: Perform batch prediction on each input file
def batch_prediction(**kwargs):
    """
    Runs batch prediction on all files in the input directory.

    Steps:
    1. Import batch prediction function from sensor pipeline.
    2. Iterate over each file in '/app/input_files'.
    3. Call 'start_batch_prediction' for each file.
    """
    try:
        from sensor.pipeline.batch_prediction import start_batch_prediction
        
        input_dir = "/app/input_files"
        # Iterate through all downloaded input files
        for file_name in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file_name)
            print(f"Running prediction on file: {file_path}")
            
            # Run prediction
            start_batch_prediction(input_file_path=file_path)
        
        print("Batch prediction completed successfully.")
    
    except Exception as e:
        print(f"Error during batch prediction: {e}")
        raise e

# Task 3: Upload prediction files to S3
def sync_prediction_dir_to_s3_bucket(**kwargs):
    """
    Uploads generated prediction output files to the configured S3 bucket.

    Steps:
    1. Fetch bucket name from environment variable.
    2. Iterate over all files in '/app/prediction' directory.
    3. Upload each file to the S3 bucket under 'prediction_files/' prefix.
    """
    try:
        bucket_name = os.getenv("BUCKET_NAME")  # S3 bucket name from env
        prediction_dir = "/app/prediction"
        
        s3_hook = S3Hook()  # Initialize S3 Hook
        
        # Iterate over prediction output files
        for file_name in os.listdir(prediction_dir):
            local_path = os.path.join(prediction_dir, file_name)
            s3_key = os.path.join("prediction_files", file_name)
            
            # Upload to S3
            s3_hook.load_file(
                filename=local_path,
                key=s3_key,
                bucket_name=bucket_name,
                replace=True
            )
            print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
    
    except Exception as e:
        print(f"Error uploading prediction files: {e}")
        raise e

# Define DAG
with DAG(
    dag_id='sensor_prediction',
    default_args={'retries': 2},  # Retry twice on failure
    description='Sensor Fault Detection Pipeline',
    schedule_interval="@weekly",  # Runs weekly
    start_date=pendulum.datetime(2025, 3, 13, tz="UTC"),
    catchup=False,  # Do not backfill old runs
    tags=['example'],
) as dag:

    # Task 1: Download input files from S3
    download_input_files = PythonOperator(
        task_id="download_files",
        python_callable=download_files
    )

    # Task 2: Generate batch predictions
    generate_prediction_files = PythonOperator(
        task_id="batch_prediction",
        python_callable=batch_prediction
    )

    # Task 3: Upload prediction output files back to S3
    upload_prediction_files = PythonOperator(
        task_id="upload_prediction_files",
        python_callable=sync_prediction_dir_to_s3_bucket
    )

    # Set task dependencies (sequential execution)
    download_input_files >> generate_prediction_files >> upload_prediction_files
