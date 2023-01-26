from imports import *

import pandas as pd
import shutil


def create_key_list(path):
    listdir = os.listdir(path)
    df_symbols = pd.DataFrame(listdir)
    return df_symbols


def move_images(old_path, new_path, df_symbols):
    if not os.path.isdir(new_path):
        os.mkdir(new_path)
    for index in df_symbols.index:
        # image_list = os.listdir(old_path+str(index))
        if not os.path.isdir(new_path + str(index)):
            shutil.copytree(old_path + str(df_symbols.iloc[index, 0]), new_path + str(index))


symbols = create_key_list(IMAGE_PATH + paths["or_train"])
# move_images(IMAGE_PATH + paths["or_train_data"], IMAGE_PATH + paths["train_data"], symbols)
move_images(IMAGE_PATH + paths["or_test"], IMAGE_PATH + paths["test"], symbols)
