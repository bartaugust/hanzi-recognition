import logging

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.utils import Sequence

from imports import *

import pandas as pd
import cv2


# Generator
class DataLoading:
    def __init__(self):
        self.dataframe = pd.DataFrame({})

    def remove_symbols(self):
        self.df.drop(self.df.index[10:], inplace=True)

    def create_dataframe_from_files(self, path):
        logging.debug('loading data')
        data_dict = {directory: os.listdir(path + directory) for directory in os.listdir(path)}

        self.df = pd.DataFrame(data_dict.items(), columns=['Symbol', 'Image'])
        self.remove_symbols()
        self.df = self.df.explode('Image').sample(frac=1).reset_index(drop=True)
        logging.debug('created dataframe')

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

        labels, images = self.load_images_from_dataframe(train_df, "train_data")
        np.save('labels.npy', labels)
        np.save('images.npy', images)

    def load_from_npy(self):
        labels = np.load('labels.npy', allow_pickle=True)
        images = np.load('images.npy')
        return labels, images

    def create_generator(self, dataset):
        config_train = {
            'rescale': 1. / 255,
            'shear_range': 0.2,
            'zoom_range': 0.2
        }

        config_test = {
            "rescale": 1. / 255
        }

        datagen = ImageDataGenerator(**config_train)

    def get_generators(self):
        pass

    def get_images(self, load_from):
        if load_from == 'numpy':
            return self.load_from_npy()
        else:
            return self.load_images_from_dataframe()


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    IMAGE_PATH + 'CASIA-HWDB_Train/Train',
    target_size=(parameters['image_params']['width'], parameters['image_params']['height']),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    IMAGE_PATH + 'CASIA-HWDB_Test/Test',
    target_size=(parameters['image_params']['width'], parameters['image_params']['height']),
    batch_size=32,
    class_mode='binary')

# class HanziDataset(Sequence):
#     def __init__(self, list_IDs, labels, image_path, mask_path,
#                  to_fit=True, batch_size=32, dim=(256, 256),
#                  n_channels=1, n_classes=10, shuffle=True):
#         self.list_IDs = list_IDs
#         self.labels = labels
#         self.image_path = image_path
#         self.mask_path = mask_path
#         self.to_fit = to_fit
#         self.batch_size = batch_size
#         self.dim = dim
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.on_epoch_end()


# save_images_to_npy()
if __name__ == '__main__':
    labels, images = load_from_npy()
    vocab = np.unique(labels)
    vocab_layer = layers.StringLookup(vocabulary=vocab)
