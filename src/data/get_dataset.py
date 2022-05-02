import argparse
from fileinput import filename
from pathlib import Path
from zipfile import ZipFile
import random
import re
import sys
#sys.path.insert(0, "./src/config")

#from config import ZIPFILE_PATH

ZIPFILE_PATH = "/content/drive/MyDrive/breast-cancer-classifier/datasets/zipfile/archive.zip"

def get_all_patient_ids():
    """
    Return list of all patient ids
    """
    all_patient_ids = set()

    with ZipFile(ZIPFILE_PATH, 'r') as zip:
        filename_list = zip.namelist()
      
    for filename in filename_list:
        if not filename.startswith('IDC_regular_ps50'):
            id = re.findall("(\A\d+)\/", filename)[0]
            all_patient_ids.add(int(id))
        else:
          break

    return all_patient_ids

def get_a_part_of_dataset(num_patients, des_dir, all_patient_ids, seed):
    """
    Get a part of the dataset 
    """
    random.seed(seed)
    a_part_of_patient_ids = random.sample(all_patient_ids, num_patients)
    with ZipFile(ZIPFILE_PATH, 'r') as zip:
        filename_list = zip.namelist()
        for id in a_part_of_patient_ids:
            for filename in filename_list:
                if filename.startswith(str(id)):
                    zip.extract(filename, des_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get a part of the orginal dataset from a number of patients")
    parser.add_argument("-n", "--num_patients", type=int, help="the number of patients, less than or equal to 279")
    parser.add_argument("-d", "--des_dir", type=Path, default="./datasets/small-dataset", help="The path to directory to store dataset")
    parser.add_argument("-s", "--seed", type=int, default=6, help="seed in random.seed()")
    args = vars(parser.parse_args())

    num_patients = args['num_patients']
    des_dir = args['des_dir']
    seed = args['seed']

    all_patient_ids = get_all_patient_ids()
    get_a_part_of_dataset(num_patients, des_dir, all_patient_ids,seed)