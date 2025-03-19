import os ,sys
import logging 
from datetime import datetime
#log file
log_file_name = f"{datetime.now().strftime('%m%d%y__%H%M%S')}.log"

#log directory
log_directory=os.path.join(os.getcwd() , "logs")

#create directory if not available
os.makedirs(log_directory, exist_ok=True)

#log file path 
log_file_path=os.path.join(log_directory ,  log_file_name)

logging.basicConfig( 

    filename= log_file_path , 
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s " , 
    level=logging.DEBUG, 

)
