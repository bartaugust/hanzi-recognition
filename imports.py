import os
from dotenv import load_dotenv
import json
import logging

load_dotenv()

PROJECT_PATH = os.getenv('PROJECT_PATH')

with open(PROJECT_PATH + 'parameters.json') as f:
    parameters = json.load(f)
    paths = parameters['paths']

logging.getLogger().setLevel(parameters['logging']['level'])