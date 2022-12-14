import logging

import numpy as np

from imports import *

import pandas as pd
import cv2

from keras import layers
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
    return label_list, np.array(image_list).astype(np.float32)/255


def save_images_to_npy():
    # test_df = create_dataframe_from_files(IMAGE_PATH+paths["test_data"])
    train_df = create_dataframe_from_files(IMAGE_PATH + paths["train_data"])

    labels, images = load_images_from_dataframe(train_df, "train_data")
    np.save('labels.npy', labels)
    np.save('images.npy', images)


def load_from_npy():
    labels = np.load('labels.npy',allow_pickle=True)
    images = np.load('images.npy')
    return labels, images



# save_images_to_npy()
labels, images = load_from_npy()
vocab = np.unique(labels)
vocab_layer = layers.StringLookup(vocabulary=vocab)


