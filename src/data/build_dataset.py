# This will build training, validation and test generator

from imutils import paths
import random, shutil, os

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import sys

sys.path.insert(0, "./src/config")
from config import *






TRAIN_SIZE = len(list(paths.list_images(TRAIN_PATH)))
VAL_SIZE = len(list(paths.list_images(VAL_PATH)))
TEST_SIZE = len(list(paths.list_images(TEST_PATH)))


def get_train_val_test_generator(train_path, val_path, test_path, batch_size=32):
    """
    Return training, validation and test generator
    """
    train_generator = ImageDataGenerator(rescale=1./255,
                                         rotation_range=20,
                                         zoom_range=0.05,
                                         width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         shear_range=0.05,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         fill_mode = "nearest" )
    val_test_generator = ImageDataGenerator(rescale=1./255)    

    train_gen = train_generator.flow_from_directory(directory=train_path, 
                                                    class_mode='categorical',
                                                    target_size=(48,48),
                                                    color_mode='rgb',
                                                    shuffle=True,
                                                    batch_size=batch_size) 


    val_gen = val_test_generator.flow_from_directory(directory=val_path,
                                                class_mode="categorical",
                                                target_size=(48, 48),
                                                color_mode="rgb",
                                                shuffle=False,
                                                batch_size=batch_size)

    test_gen = val_test_generator.flow_from_directory(directory=test_path,
                                                class_mode="categorical",
                                                target_size=(48, 48),
                                                color_mode="rgb",
                                                shuffle=False,
                                                batch_size=batch_size)


    return train_gen, val_gen, test_gen


def get_class_weight(train_path):


    train_path_list = list(paths.list_images(train_path))
    # TRAIN_PATH_SIZE = len(TRAIN_PATHS)

    train_labels = [int(p.split(os.path.sep)[-2]) for p in train_path_list]
    train_labels = to_categorical(train_labels)

    # num_examples = train_labels.shape[0]

    class_totals = np.sum(train_labels, axis=0)

    class_weight = {}
    for i in range(len(class_totals)):
        class_weight[i] = class_totals.max() / class_totals[i]  
    return class_weight

if __name__ == "__main__":
    train_gen, val_gen, test_gen = get_train_val_test_generator(TRAIN_PATH, VAL_PATH, TEST_PATH)
    print(get_class_weight(TRAIN_PATH))
    x, y = train_gen.__getitem__(0)
    plt.imshow(x[0])

