import os

class Settings():
    ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
    
settings = Settings()