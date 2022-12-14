from imports import *

import cv2
import tensorflow as tf
from keras import layers

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(parameters["image_params"]["height"], parameters["image_params"]["width"]),
    layers.Rescaling(1. / 255),
])

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.2),

])

preprocessing = tf.keras.Sequential([
    resize_and_rescale,
    data_augmentation
])
