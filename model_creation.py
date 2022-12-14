import logging

from imports import *

from data_augmentation import data_augmentation

import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint

from all_models import base_models
from data_loading import vocab_layer, labels, images



def train_model(model, model_name, train_ds):
    logging.debug('started training')
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience=parameters["model"]["es_patience"],
                       min_delta=parameters["model"]["es_min_delta"])

    model.compile(optimizer=parameters["model"]["optimizer"],
                  loss='sparse_categorical_crossentropy',
                  metrics=parameters["model"]["metrics"])
    model.build()
    # print(model.summary())
    history = model.fit(
        train_ds,
        vocab_layer(labels)-1,
        validation_split=0.2,
        epochs=parameters["model"]["epochs"],
        batch_size=parameters["model"]["batch_size"],
        callbacks=[es],
    )
    # model.save(PROJECT_PATH + paths["models"] + model_name + '/' + str(len(os.listdir(PROJECT_PATH + paths["models"]))))


def load_model(model_name, nr=-1):
    try:
        if nr == -1:
            nr = str(len(os.listdir(PROJECT_PATH + paths["models"])) - 1)
        if len(os.listdir(PROJECT_PATH + paths["models"])) == 0:
            raise Exception(f'No saved models named {model_name}')
        models.load_model(PROJECT_PATH + paths["models"] + model_name + '/' + str(nr))
    except:
        logging.exception('')


train_model(base_models['high_performance_cnn'], 'high_performance_cnn', images)
