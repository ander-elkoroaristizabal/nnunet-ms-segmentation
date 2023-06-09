"""
This script performs the stratified train-test and CV train-val splits,
and moves the test cases to the corresponding directory.
"""
import json
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from custom_scripts.A_config import (
    TRAIN_LABELS_DIR,
    TERMINATION,
    TRAIN_IMAGES_DIR,
    TEST_IMAGES_DIR,
    TEST_LABELS_DIR,
    PREPROCESSED_DATASET_DIR
)
from custom_scripts.utils import extract_id_from_image_filename, analyse_cases, list_all_nifti_files

if __name__ == '__main__':
    # Loading all ids:
    all_images = list_all_nifti_files(TRAIN_IMAGES_DIR)
    all_ids = sorted({extract_id_from_image_filename(file_name) for file_name in all_images})

    # We should do this just once, so we should have 117 images:
    try:
        assert len(all_ids) == 117
    except AssertionError:
        raise ValueError("This script is intended to be run just once!")

    # Extracting stats from all images:
    lesions_analysis = analyse_cases(ids=all_ids, labels_dir=TRAIN_LABELS_DIR)

    # Filling NAs:
    cols_w_possible_NAs = [
        'mean_basal_lesion_vol', 'median_basal_lesion_vol', 'mean_new_lesion_vol', 'median_new_lesion_vol'
    ]
    lesions_analysis[cols_w_possible_NAs] = lesions_analysis[cols_w_possible_NAs].fillna(value=0)

    # Stratification:

    # Cutting into categorical:
    lesions_analysis['bl_bin'] = pd.qcut(lesions_analysis['n_basal_lesions'], 3)  # Cutting by quantiles
    lesions_analysis['nl_bin'] = pd.cut(  # Cutting by specific boundaries
        lesions_analysis['n_new_lesions'],
        bins=[0, 1, 5, np.inf],
        right=False
    )
    # Combining both criteria:
    lesions_analysis['stratification_bin'] = (
            lesions_analysis['bl_bin'].astype(str) + ' & ' + lesions_analysis['nl_bin'].astype(str)
    )
    # Unifying "contiguous" rare class for easier splitting:
    rare_classes = ['(39.667, 75.0] & [1.0, 5.0)', '(39.667, 75.0] & [5.0, inf)']
    lesions_analysis.loc[
        lesions_analysis['stratification_bin'].isin(rare_classes), 'stratification_bin'
    ] = ' | '.join(rare_classes)
    # Factorizing:
    lesions_analysis['stratification_class'] = pd.factorize(lesions_analysis['stratification_bin'])[0]

    # Train test split:
    train_val_ids, test_ids = train_test_split(lesions_analysis['case_id'], test_size=0.3,
                                               stratify=lesions_analysis['stratification_class'])
    # Moving test images and labels:
    os.mkdir(TEST_IMAGES_DIR)
    os.mkdir(TEST_LABELS_DIR)
    for test_case in test_ids:
        # Images:
        basal_image = f"{test_case}_0000" + TERMINATION
        followup_image = f"{test_case}_0001" + TERMINATION
        shutil.move(TRAIN_IMAGES_DIR / basal_image, TEST_IMAGES_DIR / basal_image)
        shutil.move(TRAIN_IMAGES_DIR / followup_image, TEST_IMAGES_DIR / followup_image)
        # Mask:
        mask_image = test_case + TERMINATION
        shutil.move(TRAIN_LABELS_DIR / mask_image, TEST_LABELS_DIR / mask_image)

    # Cross-validation folds generation:
    train_stratification_bins = lesions_analysis.iloc[train_val_ids.index]['stratification_class']

    cv_folds = []
    skf = StratifiedKFold(n_splits=5)
    # Generation of the "splits_final.json" file:
    for train, val in skf.split(X=train_val_ids, y=train_stratification_bins):
        fold = {
            "train": train_val_ids.iloc[train].tolist(),
            "val": train_val_ids.iloc[val].tolist()
        }
        cv_folds.append(fold)
    with open(PREPROCESSED_DATASET_DIR / 'splits_final.json', 'w') as f:
        json.dump(cv_folds, f)
    print("Done!")
