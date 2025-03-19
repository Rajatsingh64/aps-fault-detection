import sys
import os

def error_message_detail(error, error_detail: sys):
    """
    Generate a detailed error message using traceback information.
    
    Parameters:
        error (Exception): The exception object that was raised.
        error_detail (module): The sys module, used to access exception details via sys.exc_info().
    
    Returns:
        str: A formatted error message including the file name, line number, and error message.
    """
    # Extract the traceback object from the exception details
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the file name where the exception occurred
    error_message = "Error occurred in Python script [{0}] at line number [{1}]: {2}".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class SensorException(Exception):
    """
    Custom exception class for sensor-related errors.
    """
    def __init__(self, error_message, error_detail: sys):
        """
        Initialize the SensorException with a detailed error message.
        
        Parameters:
            error_message (Exception): The original exception that was raised.
            error_detail (module): The sys module to access traceback information.
        """
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message
