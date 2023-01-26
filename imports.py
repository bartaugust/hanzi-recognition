import os
from dotenv import load_dotenv
import json
import logging
import numpy as np
import tensorflow as tf

# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')

physical_devices = tf.config.list_physical_devices('GPU')
logging.info(physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

load_dotenv()

PROJECT_PATH = os.getenv('PROJECT_PATH')
IMAGE_PATH = os.getenv('IMAGE_PATH')

with open(PROJECT_PATH + 'parameters.json') as f:
    parameters = json.load(f)
    paths = parameters['paths']

logging.getLogger().setLevel(parameters['logging']['level'])

# tf.keras.backend.clear_session()
# tf.config.optimizer.set_jit(False)