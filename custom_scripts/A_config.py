import os
from pathlib import Path
import pandas as pd

# Pandas config
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 900)

# General Paths
ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DATA_PATH = ROOT / "NEW_LESIONS_IMAGINEM"
NNUNET_RAW_PATH = ROOT / "data" / "nnUNet_raw_data"
NNUNET_PREPROCESSED_PATH = ROOT / "data" / "nnUNet_preprocessed_data"
NNUNET_RESULTS_PATH = ROOT / "nnUNet_results"

# Exporting general paths

os.environ['nnUNet_raw'] = str(NNUNET_RAW_PATH)
os.environ['nnUNet_preprocessed'] = str(NNUNET_PREPROCESSED_PATH)
os.environ['nnUNet_results'] = str(NNUNET_RESULTS_PATH)

# Dataset specific paths and configuration
DATASET = "Dataset100_MSSEG"
TERMINATION = ".nii.gz"
CONFIGURATION = "3d_fullres"
PLAN = 'nnUNetPlans'

RAW_DATASET_DIR = NNUNET_RAW_PATH / DATASET
PREPROCESSED_DATASET_DIR = NNUNET_PREPROCESSED_PATH / DATASET

# Split specific paths
TRAIN_IMAGES_DIR = RAW_DATASET_DIR / "imagesTr"
TRAIN_LABELS_DIR = RAW_DATASET_DIR / "labelsTr"
TEST_IMAGES_DIR = RAW_DATASET_DIR / "imagesTs"
TEST_LABELS_DIR = RAW_DATASET_DIR / "labelsTs"

# Test
NNUNET_TEST_RESULTS_PATH = ROOT / 'nnUNet_test_results'
