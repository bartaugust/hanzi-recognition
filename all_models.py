from imports import *
from data_loading import vocab
from data_augmentation import data_augmentation

import tensorflow as tf
from keras import layers, models

base_models = {
    "simple_model":
        tf.keras.Sequential([
            data_augmentation,
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((3, 3)),
            layers.Flatten(),
            # layers.Dropout(0.5),
            layers.Dense(len(vocab), activation='softmax')
        ]),

    "high_performance_cnn":
    # https://arxiv.org/pdf/1812.11489v2.pdf
        tf.keras.Sequential([
            data_augmentation,
            layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1)),
            layers.BatchNormalization(),
            layers.AveragePooling2D((3, 3), strides=(1, 1)),
            layers.Conv2D(96, (3, 3), activation='relu', strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(96, (3, 3), activation='relu', strides=(1, 1)),
            layers.BatchNormalization(),
            layers.AveragePooling2D((3, 3), strides=(1, 1)),
            layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(96, (3, 3), activation='relu', strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1)),
            layers.BatchNormalization(),
            layers.AveragePooling2D((3, 3), strides=(1, 1)),
            layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1)),
            layers.BatchNormalization(),
            layers.AveragePooling2D((3, 3), strides=(1, 1)),
            layers.Conv2D(448, (3, 3), activation='relu', strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(448, (3, 3), activation='relu', strides=(1, 1)),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(len(vocab), activation='softmax')
        ]),

}
