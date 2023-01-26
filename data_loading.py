import logging
import os

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from tensorflow.keras.utils import Sequence
# import tfds

from imports import *

import pandas as pd
import cv2


# Generator
class DataLoading:
    def __init__(self):
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.df = None
        self.config = {
            'train': {
                'rescale': 1. / 255,
                # 'shear_range': 0.2,
                # 'zoom_range': 0.2,
                'validation_split': 0.2
            },
            'test': {
                "rescale": 1. / 255
            }}

    def remove_symbols(self):
        self.df.drop(self.df.index[10:], inplace=True)

    def create_dataframe_from_files(self, path):
        logging.debug('loading data')
        data_dict = {directory: os.listdir(path + directory) for directory in os.listdir(path)}

        self.df = pd.DataFrame(data_dict.items(), columns=['Symbol', 'Image'])
        self.remove_symbols()
        self.df = self.df.explode('Image').sample(frac=1).reset_index(drop=True)
        logging.debug('created dataframe')
        print('a')

    def load_images_from_dataframe(self, data_type):
        image_list = []
        label_list = self.df["Symbol"].to_numpy()
        image_paths = IMAGE_PATH + paths[data_type] + self.df["Symbol"].to_numpy() + '/' + self.df["Image"].to_numpy()
        for image_path in image_paths:
            image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_list.append(
                cv2.resize(image, (parameters["image_params"]["height"], parameters["image_params"]["width"])))
            logging.debug(f'loaded images {len(image_list)}/{len(image_paths)}')
            logging.debug(f'loaded {image_path}')
        logging.info(f'loaded all {data_type}')
        return label_list, np.array(image_list).astype(np.float32) / 255

    def save_images_to_npy(self):
        # test_df = create_dataframe_from_files(IMAGE_PATH+paths["test_data"])
        train_df = self.create_dataframe_from_files(IMAGE_PATH + paths["train_data"])

        labels, images = self.load_images_from_dataframe("train_data")
        np.save('labels.npy', labels)
        np.save('images.npy', images)

    def load_from_npy(self):
        labels = np.load('labels.npy', allow_pickle=True)
        images = np.load('images.npy')
        return labels, images

    def create_train_test_tf_datasets(self):
        self.train_ds = tf.data.Dataset.from_generator(lambda: self.train_generator,
                                                       output_types=(tf.float32, tf.float32),
                                                       output_shapes=([32,
                                                                       parameters['image_params']['width'],
                                                                       parameters['image_params']['height'], 3],
                                                                      [32, ]))
        self.val_ds = tf.data.Dataset.from_generator(lambda: self.val_generator,
                                                     output_types=(tf.float32, tf.float32),
                                                     output_shapes=([32,
                                                                     parameters['image_params']['width'],
                                                                     parameters['image_params']['height'], 3],
                                                                    [32, ]))
        self.test_ds = tf.data.Dataset.from_generator(lambda: self.test_generator,
                                                      output_types=(tf.float32, tf.float32),
                                                      output_shapes=([32,
                                                                      parameters['image_params']['width'],
                                                                      parameters['image_params']['height'], 3],
                                                                     [32, ]))

    def get_tf_datasets(self):
        return self.train_ds, self.val_ds, self.test_ds

    def create_train_test_generators(self):
        logging.info('creating_generators')
        train_datagen = ImageDataGenerator(**self.config['train'])
        test_datagen = ImageDataGenerator(**self.config['test'])

        self.train_generator = train_datagen.flow_from_directory(
            IMAGE_PATH + paths['train'],
            target_size=(parameters['image_params']['width'], parameters['image_params']['height']),
            batch_size=32,
            class_mode='binary',
            subset='training'
        )
        self.val_generator = train_datagen.flow_from_directory(
            IMAGE_PATH + paths['train'],
            target_size=(parameters['image_params']['width'], parameters['image_params']['height']),
            batch_size=32,
            class_mode='binary',
            subset='validation'
        )
        self.test_generator = test_datagen.flow_from_directory(
            IMAGE_PATH + paths['test'],
            target_size=(parameters['image_params']['width'], parameters['image_params']['height']),
            batch_size=32,
            class_mode='binary'
        )
        logging.info('generators_created')

    def get_generators(self):
        return self.train_generator, self.val_generator, self.test_generator

    def get_images(self, load_from='nump'):
        if load_from == 'numpy':
            return self.load_from_npy()
        else:
            return self.load_images_from_dataframe()


# vocab = os.listdir(IMAGE_PATH + paths["train"])

# save_images_to_npy()
if __name__ == '__main__':
    d = DataLoading()
    d.create_train_test_generators()
    train_generator, val_generator, test_generator = d.get_generators()

    d.create_train_test_tf_datasets()
    train_ds, val_ds, test_ds = d.get_tf_datasets()
# labels, images = load_from_npy()
# vocab = np.unique(labels)
# vocab_layer = layers.StringLookup(vocabulary=vocab)
