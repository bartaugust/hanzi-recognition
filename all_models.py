# from imports import *
# from data_loading import vocab
# from data_augmentation import data_augmentation
#
# import tensorflow as tf
# from keras import layers, models
#
# base_models = {
#     "simple_model":
#         tf.keras.Sequential([
#             data_augmentation,
#             layers.Conv2D(64, (3, 3), activation='relu'),
#             layers.Conv2D(64, (3, 3), activation='relu'),
#             layers.MaxPooling2D((3, 3)),
#             layers.Flatten(),
#             # layers.Dropout(0.5),
#             layers.Dense(len(vocab), activation='softmax')
#         ]),
#
# }

from imports import *
from data_loading import *

from keras.datasets import cifar10
import numpy as np

import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint

from abc import abstractmethod


class ModelTemplate:

    def __init__(self, all_classes):
        self.all_classes = np.arange(all_classes)
        self.train_data = train_data,
        self.test_data = test_data
        self.base_models = {
            "simple_model":
                tf.keras.Sequential([
                    # preprocessing,
                    layers.Conv2D(64, (3, 3), activation='relu',
                                  # input_shape=(
                                  #     parameters["image_params"]["height"], parameters["image_params"]["width"], 3)
                                  ),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.MaxPooling2D((3, 3)),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.MaxPooling2D((3, 3)),
                    layers.Flatten(),
                    layers.Dropout(0.3),
                    layers.Dense(1000, activation='relu')

                    # layers.Dense(len(vocab), activation='softmax')
                ]),
            "simple_model_2":
                tf.keras.models.Sequential([
                    # preprocessing,
                    layers.Conv2D(16, (3, 3), activation='relu',
                                  input_shape=(
                                      parameters["image_params"]["height"], parameters["image_params"]["width"], 3),
                                  padding='same'
                                  ),
                    layers.Conv2D(32, (3, 3),
                                  activation='relu',
                                  padding='same'),
                    layers.Conv2D(64, (3, 3),
                                  activation='relu',
                                  padding='same'),
                    layers.MaxPooling2D(2, 2),
                    layers.Conv2D(128, (3, 3),
                                  activation='relu',
                                  padding='same'),

                    layers.Flatten(),
                    layers.Dense(256, activation='relu'),
                    layers.Dropout(0.3),
                    layers.BatchNormalization(),
                    layers.Dense(256, activation='relu'),
                    layers.Dropout(0.3),
                    layers.BatchNormalization(),
                ]),
            "high_performance_cnn":
            # https://arxiv.org/pdf/1812.11489v2.pdf
                tf.keras.Sequential([
                    # data_augmentation,
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
        self.model = self.base_models["high_performance_cnn"]
        es = EarlyStopping(monitor='val_loss',
                           mode='min',
                           verbose=1,
                           patience=parameters["model"]["es_patience"],
                           min_delta=parameters["model"]["es_min_delta"])
        self.callbacks = [es]
        self.history = {}

    def encode_classes(self, classes):
        i = 0
        encoding = -np.ones(np.max(classes) + 1).astype('int8')
        for elem in classes:
            encoding[elem] = i
            i += 1
        return encoding

    def divide_data(self, classes, data_type):
        if data_type == "test_data":
            data = self.test_data
        else:
            data = self.train_data
        selected_indexes = np.argwhere(np.isin(data[1], classes))
        selected_indexes = selected_indexes[:, 0]
        selected_images = data[0][selected_indexes]
        selected_labels = data[1][selected_indexes]
        return (selected_images, selected_labels)

    def compile_model(self, model):

        model.compile(optimizer=parameters["model"]["optimizer"],
                      # loss=tf.keras.losses.CategoricalCrossentropy(),
                      loss='sparse_categorical_crossentropy',
                      metrics=parameters["model"]["metrics"]
                      )
        model.build()

    @abstractmethod
    def train_groups(self, group_slice=-1):
        pass

    @abstractmethod
    def predict_all(self, images):
        pass

    def train_model(self,  model,  train_ds):
        logging.debug('started training')

        # print(model.summary())
        self.history = model.fit(
            train_ds,
            vocab_layer(labels)-1,
            validation_split=0.2,
            epochs=parameters["model"]["epochs"],
            batch_size=parameters["model"]["batch_size"],
            callbacks=self.callbacks,
        )

