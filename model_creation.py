from imports import *

from data_augmentation import resize_and_rescale, data_augmentation

import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint



def train_model(model, model_name, train_ds, val_ds):
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience=parameters["model"]["es_patience"],
                       min_delta=parameters["model"]["es_min_delta"])

    model.compile(optimizer=parameters["model"]["optimizer"],
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=parameters["model"]["metrics"])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=parameters["model"]["epochs"],
        callbacks=[es],
    )
    model.save(PROJECT_PATH + paths["models"] + model_name + '/' + str(len(os.listdir(PROJECT_PATH + paths["models"]))))


def load_model(model_name, nr=-1):
    try:
        if nr == -1:
            nr = str(len(os.listdir(PROJECT_PATH + paths["models"])) - 1)
        if len(os.listdir(PROJECT_PATH + paths["models"])) == 0:
            raise Exception(f'No saved models named {model_name}')
        models.load_model(PROJECT_PATH + paths["models"] + model_name + '/' + str(nr))
    except:
        logging.exception('')
