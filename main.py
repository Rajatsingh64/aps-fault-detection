
from sensor.pipeline.training_pipeline import start_training_pipeline 
from sensor.pipeline.batch_prediction import start_batch_prediction 
from sensor.exception import SensorException
import os,sys

file_path =r"aps_failure_training_set1.csv"

if __name__=="__main__":
    try :
        
        training_output= start_training_pipeline()
        Batch_output=start_batch_prediction(input_file_path=file_path)
        print(">"*15," Current Prediction is " , ">"*15)
        print(Batch_output)
    

        
        
    except Exception as e:
        raise SensorException(e , sys)
