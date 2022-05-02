# This will build training, validation and test generator
import argparse
from cgi import test
from pathlib import Path
from imutils import paths
import random, shutil, os

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

#import sys
#sys.path.insert(0, "./src/config")
#from config import *



def split_data(input_dataset, train_path, val_path, test_path, train_split, val_split):
    """
    Split dataset into training, validation and test sets
    Args:
        input_dataset (string): path to dataset
        train_split (float): proportion of training test size in dataset
        val_split (float): proportion of validation test size in training set
    Returns:
        datasets: (list): 3 tuples, each with information required to organize all image paths into training, validation and test data.
    """
    original_path_list = list(paths.list_images(input_dataset))

    random.seed(7)
    random.shuffle(original_path_list)

    index = int(len(original_path_list) * train_split)
    train_path_list = original_path_list[:index]
    test_path_list = original_path_list[index:]

    index = int(len(train_path_list) * val_split)
    val_path_list = train_path_list[:index]
    train_path_list = train_path_list[index:]

    datasets = [("training", train_path_list, train_path),
                ("validation", val_path_list, val_path),
                ("test", test_path_list, test_path)]


    for (set_type, original_path, base_path) in datasets:
        print(f'Building {set_type} set')

        if not os.path.exists(base_path):
            print(f'Building directory {base_path}')
            os.makedirs(base_path)

        for path in original_path:
            file = path.split(os.path.sep)[-1]
            label = file[-5:-4]

            label_path = os.path.sep.join([base_path, label])
            if not os.path.exists(label_path):
                print(f'Building directory {label_path}')
                os.makedirs(label_path)

            new_path = os.path.sep.join([label_path, file])
            shutil.copy2(path, new_path)

    train_size, val_size, test_size = get_train_val_test_size(train_path, val_path, test_path)
    print(f"\nThe sizes of training set, validation set and test set is {train_size}, {val_size}, {test_size} images, repsectively")

    return datasets

def get_train_val_test_size(train_path, val_path, test_path):
    """
    Return the size of training, validation and test sets
    """
    train_size = len(list(paths.list_images(train_path)))
    val_size = len(list(paths.list_images(val_path)))
    test_size = len(list(paths.list_images(test_path)))
    return train_size, val_size, test_size

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
    
    parser = argparse.ArgumentParser(description="Split data into training, validation and test set")
    parser.add_argument("--src_dir", type=Path, help="the path to the directory that contains the dataset splited")
    parser.add_argument("--des_dir", type=Path, help="the path to the directory that store the dataset after split")
    parser.add_argument("--train_split",  type=float, default=0.8, help="proportion of training set (included validation set) in dataset")
    parser.add_argument("--val_split", type=float, default=0.1, help="proportion of validation set in training set")
    args = vars(parser.parse_args())

    src_dir = str(args['src_dir'])
    des_dir = str(args['des_dir'])
    train_split = args['train_split']
    val_split = args['val_split']

    input_dataset = src_dir
    train_path = os.path.sep.join([des_dir, "training"])
    val_path = os.path.sep.join([des_dir, 'validation'])
    test_path = os.path.sep.join([des_dir, "test"])



    """original_path_list = list(paths.list_images(input_dataset))

    random.seed(7)
    random.shuffle(original_path_list)

    index = int(len(original_path_list) * train_split)
    train_path_list = original_path_list[:index]
    test_path_list = original_path_list[index:]

    index = int(len(train_path_list) * val_split)
    val_path_list = train_path_list[:index]
    train_path_list = train_path_list[index:]

    datasets = [("training", train_path_list, train_path),
                ("validation", val_path_list, val_path),
                ("test", test_path_list, test_path)]


    for (set_type, original_path, base_path) in datasets:
        print(f'Building {set_type} set')

        if not os.path.exists(base_path):
            print(f'Building directory {base_path}')
            os.makedirs(base_path)

        for path in original_path:
            file = path.split(os.path.sep)[-1]
            label = file[-5:-4]

            label_path = os.path.sep.join([base_path, label])
            if not os.path.exists(label_path):
                print(f'Building directory {label_path}')
                os.makedirs(label_path)

            new_path = os.path.sep.join([label_path, file])
            shutil.copy2(path, new_path)

    train_size, val_size, test_size = get_train_val_test_size(train_path, val_path, test_path)
    print(f"\nThe sizes of training set, validation set and test set is {train_size}, {val_size}, {test_size} images, repsectively")
    """
    datastet = split_data(input_dataset, train_path, val_path, test_path, train_split, val_split)

    #train_gen, val_gen, test_gen = get_train_val_test_generator(TRAIN_PATH, VAL_PATH, TEST_PATH)
    #print(get_class_weight(TRAIN_PATH))
    #x, y = train_gen.__getitem__(0)
    #plt.imshow(x[0])

