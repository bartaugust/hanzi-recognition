import logging

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.utils import Sequence

from imports import *

import pandas as pd
import cv2


def remove_symbols(df):
    return df.drop(df.index[10:])


def create_dataframe_from_files(path):
    logging.debug('loading data')
    data_dict = {directory: os.listdir(path + directory) for directory in os.listdir(path)}

    df = pd.DataFrame(data_dict.items(), columns=['Symbol', 'Image'])
    df = remove_symbols(df)
    dfe = df.explode('Image').sample(frac=1).reset_index(drop=True)
    logging.debug('created dataframe')
    return dfe


def load_images_from_dataframe(df, data_type):
    image_list = []
    label_list = df["Symbol"].to_numpy()
    image_paths = IMAGE_PATH + paths[data_type] + df["Symbol"].to_numpy() + '/' + df["Image"].to_numpy()
    for image_path in image_paths:
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_list.append(
            cv2.resize(image, (parameters["image_params"]["height"], parameters["image_params"]["width"])))
        logging.debug(f'loaded images {len(image_list)}/{len(image_paths)}')
        logging.debug(f'loaded {image_path}')
    logging.info(f'loaded all {data_type}')
    return label_list, np.array(image_list).astype(np.float32) / 255


def save_images_to_npy():
    # test_df = create_dataframe_from_files(IMAGE_PATH+paths["test_data"])
    train_df = create_dataframe_from_files(IMAGE_PATH + paths["train_data"])

    labels, images = load_images_from_dataframe(train_df, "train_data")
    np.save('labels.npy', labels)
    np.save('images.npy', images)


def load_from_npy():
    labels = np.load('labels.npy', allow_pickle=True)
    images = np.load('images.npy')
    return labels, images


# Generator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    IMAGE_PATH + 'CASIA-HWDB_Train/Train',
    target_size=(parameters['image_params']['width'],parameters['image_params']['height']),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    IMAGE_PATH +'CASIA-HWDB_Test/Test',
    target_size=(parameters['image_params']['width'],parameters['image_params']['height']),
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
