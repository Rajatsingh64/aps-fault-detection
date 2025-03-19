import pymongo
import pandas
import json
from sensor.config import mongo_client  # Importing the mongo_client from the sensor configuration module

# Define constants for database configuration and file path
database_name = "aps"  # Name of the MongoDB database
file_path = "aps_failure_training_set1.csv"  # Path to the CSV file containing training data
collection_name = "sensor"  # Name of the MongoDB collection

if __name__ == "__main__":
    # Read the CSV file into a pandas DataFrame
    df = pandas.read_csv(file_path)
    print(f"Rows and Columns: {df.shape}")

    # Reset the DataFrame index to ensure proper JSON conversion, dropping the old index
    df.reset_index(drop=True, inplace=True)

    # Convert the DataFrame to JSON records suitable for MongoDB insertion
    # The DataFrame is transposed, converted to JSON, then loaded into a list of dictionaries
    json_records = list(json.loads(df.T.to_json()).values())
    print(json_records[0])

    # Insert the JSON records into the specified MongoDB collection
    mongo_client[database_name][collection_name].insert_many(json_records)
