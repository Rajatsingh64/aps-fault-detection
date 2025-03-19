from setuptools import find_packages, setup
from typing import List

# Constants for the requirements file and the editable install option
REQUIREMENT_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."

def get_requirements() -> List[str]:
    """
    Reads the list of requirements from the requirements.txt file.
    
    This function performs the following steps:
    1. Opens the file defined by REQUIREMENT_FILE_NAME.
    2. Reads all lines from the file.
    3. Removes any newline characters from each line.
    4. Removes the editable install option (if present) from the list.
    
    Returns:
        A list of package requirement strings.
    """
    # Open and read all lines from the requirements file
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
    
    # Remove newline characters from each requirement
    requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]
    
    # Remove the editable install option if it exists in the list
    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)
    
    return requirement_list

# Package setup configuration
setup(
    name="sensor",                      # Name of the package
    version="0.0.2",                    # Package version
    author="ineuron",                   # Author of the package
    author_email="rajat.k.singh64@gmail.com",  # Author email
    packages=find_packages(),           # Automatically find packages in the project
    install_requires=get_requirements(),# List of dependencies from the requirements file
)
