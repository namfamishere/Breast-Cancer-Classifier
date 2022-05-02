import os


INPUT_DATASET = "../IDC datasets/datasets/original"
SMALL_INPUT_DATASET = "./datasets/original-small-01"
BASE_PATH = "datasets/splitted-data-01"
ZIPFILE_PATH = "datasets/zipfile/archive.zip"

TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, 'validation'])
TEST_PATH = os.path.sep.join([BASE_PATH, "test"])

TRAIN_SPLIT = 0.8 # 80% then entire dataset is used for training
VAL_SPLIT = 0.1 # 10% the entire dataset is used for validation 

NUM_EPOCHS = 40
INIT_LEARNING_RATE = 1e-2
BATCH_SIZE = 32