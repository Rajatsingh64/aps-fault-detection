import pymongo
import pandas
import json
#connecting to a MongoDB Atlas using url

#client = pymongo.MongoClient("mongodb+srv://rajatksingh64:<passwordpls>@cluster0.owfzau8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

database_name="aps"
file_path=r"D:\Downloaded files\aps_failure_training_set1.csv"
collection_name="sensor"

if __name__=="__main__":
   df=pandas.read_csv(file_path)
   print(f"Rows and Columns : {df.shape}")

   #converting dataframe into a json format ,  so we can dump it into a MongoDB
   df.reset_index(drop=True , inplace=True)
   json_records= list(json.loads(df.T.to_json()).values())
   print(json_records[0])
   #inserting  data into a mongo db
  # client[database_name][collection_name].insert_many(json_records)
   #print("records inerted properly into a mongo db")
