from setuptools import setup, find_packages  

hyphen_e_dot = '-e .'  ## This is a common entry in requirements.txt when the package is being developed locally. It tells pip to install the package in editable mode, which allows for changes to be reflected without reinstalling. However, it should not be included in the install_requires list for setup.py, as it is not a valid package requirement.

req = [] ## list for adding requirements from requirements.txt file

def read_requirements():
    with open('requirements.txt') as f:
        content = f.read().splitlines()
        req.extend(content) ## adding the content of requirements.txt to req list
   
    if hyphen_e_dot in req:
        req.remove(hyphen_e_dot) ## removing the '-e .' entry from the req list if it exists, as it is not a valid package requirement for setup.py
    return req    


set_ = setup(
    name = 'V1_app',
    version = '0.0.1',
    packages = find_packages(),
    install_requires = read_requirements(),
    description = 'A machine learning application for predicting outcomes based on various algorithms.',
    author = 'Nitishkumar Khavekar',
    author_email = 'khavekarnitishkumar@gmail.com', 
    url = 'https://github.com/Nitishkumarkhavekar/SML_Project/tree/main')