"""
This script converts the IMAGINEM dataset as downloaded from the FTP server
to the format required by nnU-Net v2, detailed in 'documentation/dataset_format.md'.
"""
import json
import os
import shutil

from custom_scripts.A_config import (
    ORIGINAL_DATA_PATH,
    RAW_DATASET_DIR,
    TRAIN_IMAGES_DIR,
    TRAIN_LABELS_DIR
)
from custom_scripts.utils import extract_id_from_image_filename, list_all_nifti_files

if __name__ == "__main__":
    warning = "This script should only be ran once. " \
              "Are you sure you want to run it? [Y]/N"
    if input(warning) != "Y":
        exit()

    # Create directories:
    try:
        os.mkdir(RAW_DATASET_DIR)
        os.mkdir(TRAIN_IMAGES_DIR)
        os.mkdir(TRAIN_LABELS_DIR)
    except FileExistsError:
        pass

    # Writing dataset JSON:
    dataset_json = {
        "channel_names": {
            "0": "Baseline",
            "1": "Follow-up"
        },
        "labels": {
            "background": 0,
            "Basal-lesions": 1,
            "New-lesions": 2
        },
        "numTraining": 81,
        "file_ending": ".nii.gz"
    }
    with open(RAW_DATASET_DIR / 'dataset.json', 'w') as f:
        json.dump(dataset_json, f)

    # Renaming of files:
    # Dividing files into masks and images:
    all_files_in_origin = list_all_nifti_files(ORIGINAL_DATA_PATH)
    mask_files = [file_name for file_name in all_files_in_origin if "mask" in file_name]
    raw_files = [file_name for file_name in all_files_in_origin if "mask" not in file_name]

    # Extracting ids:
    ids = [extract_id_from_image_filename(file_name) for file_name in raw_files]

    # Updating labels' names to nnU-Net format:
    label2newlabel = {
        label: label.split(".")[0][:-9] + '_01.nii.gz' for label in mask_files
    }
    for old_name, new_name in label2newlabel.items():
        os.rename(ORIGINAL_DATA_PATH / old_name, ORIGINAL_DATA_PATH / new_name)

    # Moving all files to the corresponding directory:
    for image in raw_files:
        shutil.move(ORIGINAL_DATA_PATH / image, TRAIN_IMAGES_DIR / image)
    for mask in label2newlabel.values():
        shutil.move(ORIGINAL_DATA_PATH / mask, TRAIN_LABELS_DIR / mask)
