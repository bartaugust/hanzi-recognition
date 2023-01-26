from datetime import datetime

import matplotlib.pyplot as plt

from data_loading import *

import numpy as np
import seaborn as sn
import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint

from data_loading import DataLoading

physical_devices = tf.config.list_physical_devices('GPU')
logging.info(physical_devices)
# tf.config.set_visible_devices([], 'CPU')
# vocab = np.unique(labels)
# vocab_layer = layers.StringLookup(vocabulary=vocab)
class ModelTemplate:

    def __init__(self, data_type='generator'):
        self.all_classes = np.array(os.listdir(IMAGE_PATH + paths["train"]))
        d = DataLoading()
        if data_type == 'generator':
            d.create_train_test_generators()
            self.train_data, self.val_data, self.test_data = d.get_generators()
        elif data_type == 'tf_data':
            d.create_train_test_generators()
            d.create_train_test_tf_datasets()
            self.train_data, self.val_data, self.test_data = d.get_tf_datasets()
        else:
            # self.train_data, self.test_data = d.load_from_npy()
            self.train_data = d.load_from_npy()
        # self.models = {}
        # self.groups =
        self.base_models = {
            "simple_model":
                tf.keras.models.Sequential([
                    # preprocessing,
                    layers.Conv2D(16, (3, 3), activation='relu',
                                  input_shape=(
                                      parameters["image_params"]["height"], parameters["image_params"]["width"], 3),
                                  padding='same'
                                  ),

                    layers.BatchNormalization(),
                    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
                    layers.BatchNormalization(),
                    layers.MaxPooling2D((2, 2), padding='same'),
                    layers.BatchNormalization(),
                    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
                    layers.BatchNormalization(),
                    layers.MaxPooling2D((2, 2), padding='same'),
                    layers.BatchNormalization(),
                    layers.GlobalAveragePooling2D(),
                    layers.Dropout(0.5),
                    layers.Dense(len(self.all_classes), activation='softmax')
                ]),
            # https://arxiv.org/pdf/1812.11489v2.pdf

        }

        self.model = self.base_models["simple_model"]

        def add_model(self, classes):
            model = models.clone_model(self.model)
            model.add(layers.Dense(len(classes), activation='softmax'))
            self.compile_model(model)

            self.models[str(classes)] = model

        logs = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        tb = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                            histogram_freq=1,
                                            )

        es = EarlyStopping(monitor='val_loss',
                           mode='min',
                           verbose=1,
                           patience=parameters["model"]["es_patience"],
                           min_delta=parameters["model"]["es_min_delta"])
        self.callbacks = [es, tb]
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

    # def train_groups(self, group_slice=-1):
    #     for classes in self.groups[:group_slice]:
    #         self.add_model(classes, parameters["model_type"])
    #         divided_data = self.divide_data(classes, "train_data")
    #         self.train_model(self.models[str(classes)], divided_data, classes)

    def compile_model(self):

        self.model.compile(optimizer=parameters["model"]["optimizer"],
                           loss='sparse_categorical_crossentropy',
                           metrics=parameters["model"]["metrics"],
                           # jit_compile=True
                           )
        self.model.build()

    def train_model(self):
        logging.debug('started training')

        # print(self.model.summary())
        self.history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            # vocab_layer(labels)-1,
            # validation_split=0.2,
            epochs=parameters["model"]["epochs"],
            batch_size=parameters["model"]["batch_size"],
            callbacks=self.callbacks,
            steps_per_epoch=14830  // 32,
            validation_steps=3160 // 32,
            # steps_per_epoch=2580015 // parameters["model"]["batch_size"],
            # validation_steps=643028 // parameters["model"]["batch_size"],
        )
        # self.history = self.model.fit(
        #     self.train_data[0],
        #     self.train_data[1] - 1,
        #     # validation_data=self.val_data,
        #     #
        #     validation_split=0.2,
        #     epochs=parameters["model"]["epochs"],
        #     batch_size=parameters["model"]["batch_size"],
        #     callbacks=self.callbacks,
        #     # steps_per_epoch=1e2,
        #     # validation_steps=1e1,
        #     # steps_per_epoch=2580015 // parameters["model"]["batch_size"],
        #     # validation_steps=643028 // parameters["model"]["batch_size"],
        # )

    def show_accuracy_loss_plots(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.show()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.show()

    def show_confusion_matrix(self):
        y_pred_proba = self.model.predict(self.test_data)
        y_true = self.test_data.classes
        y_pred = np.argmax(y_pred_proba, axis=1).reshape(-1, )
        # y_pred_proba = self.model.predict(self.test_data)
        # y_pred = np.argmax(y_pred_proba, axis=1).reshape(-1, )

        cm = tf.math.confusion_matrix(y_true, y_pred)

        df_cm = pd.DataFrame(cm, index=self.all_classes,
                             columns=self.all_classes)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm)
        plt.show()


if __name__ == '__main__':
    md = ModelTemplate()
    md.compile_model()
    md.train_model()
