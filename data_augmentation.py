from imports import *

import cv2
import tensorflow as tf
from keras import layers

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.2, input_shape=(parameters["image_params"]["height"], parameters["image_params"]["width"],3)),

])
