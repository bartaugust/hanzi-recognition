import os
from dotenv import load_dotenv
import json
import logging
import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

load_dotenv()

PROJECT_PATH = os.getenv('PROJECT_PATH')
IMAGE_PATH = os.getenv('IMAGE_PATH')

with open(PROJECT_PATH + 'parameters.json') as f:
    parameters = json.load(f)
    paths = parameters['paths']

logging.getLogger().setLevel(parameters['logging']['level'])