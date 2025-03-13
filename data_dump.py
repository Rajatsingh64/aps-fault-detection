import pymongo
import pandas
import json
from sensor.config import mongo_client
#connecting to a MongoDB Atlas using url

database_name="aps"
file_path="aps_failure_training_set1.csv"
collection_name="sensor"

if __name__=="__main__":
   df=pandas.read_csv(file_path)
   print(f"Rows and Columns : {df.shape}")

   #converting dataframe into a json format ,  so we can dump it into a MongoDB
   df.reset_index(drop=True , inplace=True)
   json_records= list(json.loads(df.T.to_json()).values())
   print(json_records[0])
   #inserting  data into a mongo db
   mongo_client[database_name][collection_name].insert_many(json_records)
   