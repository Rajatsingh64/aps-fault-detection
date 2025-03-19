# ============================================================
# IMPORTS
# ============================================================
import pymongo
import pandas
import json
import os
from dataclasses import dataclass

# ============================================================
# ENVIRONMENT VARIABLES SETUP
# ============================================================
@dataclass
class Environment_variable:
    """
    Dataclass for holding environment variables.
    """
    pymongo_url: str = os.getenv("MONGO_DB_URL")

# Instantiate the environment variable class
env_var = Environment_variable()

# ============================================================
# MONGO CLIENT SETUP
# ============================================================
# Create a MongoDB client using the URL from environment variables
mongo_client = pymongo.MongoClient(host=env_var.pymongo_url)

# ============================================================
# CONSTANTS
# ============================================================
# Target column for model predictions or data processing
TARGET_COLUMN = "class"
