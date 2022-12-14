from imports import *

import pandas as pd
import cv2



def remove_symbols(data_dict):
    return data_dict


def create_dataframe_from_files(path):
    data_dict = {directory: os.listdir(path + directory) for directory in os.listdir(path)}
    data_dict = remove_symbols(data_dict)
    df = pd.DataFrame(data_dict.items(),columns=['Symbol', 'Image'])
    return df.explode('Image').sample(frac=1).reset_index(drop=True)


def load_images_from_dataframe(df,data_type):
    image_list = []
    label_list = df["Symbol"].to_numpy()
    image_paths = IMAGE_PATH+paths[data_type]+df["Symbol"].to_numpy()+'/'+df["Image"].to_numpy()
    for image_path in image_paths:
        image_list.append(cv2.imread(image_path))
    return label_list, image_list


test_df = create_dataframe_from_files(IMAGE_PATH+paths["test_data"])
train_df = create_dataframe_from_files(IMAGE_PATH+paths["train_data"])

labels, images = load_images_from_dataframe(test_df,"test_data")
