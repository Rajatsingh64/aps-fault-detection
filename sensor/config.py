import pymongo 
import pandas 
from dataclasses import dataclass 
import json 
import os



@dataclass 
class Environment_variable() :
    pymongo_url :str = os.getenv("MONGO_DB_URL")


env_var = Environment_variable()
mongo_client = pymongo.MongoClient(host=env_var.pymongo_url)
TARGET_COLUMN="class"

