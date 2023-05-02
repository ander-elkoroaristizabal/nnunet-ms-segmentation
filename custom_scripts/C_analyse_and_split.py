import json
import os
import pathlib
import shutil
from typing import List

import nibabel as nib
import numpy as np
from scipy.ndimage import label
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split

from custom_scripts.A_config import (
    TRAIN_LABELS_DIR,
    TERMINATION,
    TRAIN_IMAGES_DIR,
    TEST_IMAGES_DIR,
    TEST_LABELS_DIR,
    PREPROCESSED_DATASET_DIR
)


def detect_lesions(mask: nib.nifti1.Nifti1Image):
    """Detect and label different lesions by using a pattern.

    Args:
        mask: nifti image with the mask

    Returns:
        basal_lesions_map: image with a different label (int) for each lesion
        joint_lesions: a dict with basal and new lesion identifiers
    """
    # Detecting lesions:
    basal_lesions_map, n_basal_lesions = label((mask.get_fdata() == 1).astype(int))
    followup_lesions_map, n_followup_lesions = label((mask.get_fdata() == 2).astype(int))
    # Merging results:
    new_lesion_ids = []
    for new_lesion in range(1, n_followup_lesions + 1):
        new_lesion_id = new_lesion + n_basal_lesions
        basal_lesions_map[followup_lesions_map == new_lesion] = new_lesion_id
        new_lesion_ids.append(new_lesion_id)
    joint_lesions = {'basal': list(range(1, n_basal_lesions + 1)), 'new': new_lesion_ids}
    return basal_lesions_map, joint_lesions


def analyse_cases(ids: List[str], labels_dir: pathlib.Path):
    analysis_results = []
    for case_id in tqdm(ids):
        case_mask = nib.load(labels_dir / (case_id + TERMINATION))
        lesions_map, lesions = detect_lesions(case_mask)
        total_basal_lesion_vol = (case_mask.get_fdata() == 1).sum()
        total_new_lesion_vol = (case_mask.get_fdata() == 2).sum()
        mean_basal_lesion_vol = np.median([(lesions_map == b_lesion).sum() for b_lesion in lesions['basal']])
        median_basal_lesion_vol = np.mean([(lesions_map == b_lesion).sum() for b_lesion in lesions['basal']])
        mean_new_lesion_vol = np.median([(lesions_map == b_lesion).sum() for b_lesion in lesions['new']])
        median_new_lesion_vol = np.mean([(lesions_map == b_lesion).sum() for b_lesion in lesions['new']])
        case_results = {
            "case_id": case_id,
            "n_lesions": len(lesions['basal']) + len(lesions['new']),
            "n_basal_lesions": len(lesions['basal']),
            "n_new_lesions": len(lesions['new']),
            "mean_basal_lesion_vol": mean_basal_lesion_vol,
            "median_basal_lesion_vol": median_basal_lesion_vol,
            "total_basal_lesion_vol": total_basal_lesion_vol,
            "mean_new_lesion_vol": mean_new_lesion_vol,
            "median_new_lesion_vol": median_new_lesion_vol,
            "total_new_lesion_vol": total_new_lesion_vol
        }
        analysis_results.append(case_results)
    return pd.DataFrame.from_records(analysis_results)


if __name__ == '__main__':
    # Loading ids:

    all_images = os.listdir(TRAIN_IMAGES_DIR)
    all_ids = sorted({file_name.split(".")[0][:-5] for file_name in all_images})

    # We should do this just once, so we should have 117 images:
    try:
        assert len(all_ids) == 117
    except AssertionError:
        raise ValueError("This script is intended to be run just once!")

    # TODO ander 12/4/23: Why does measuring volume on the lesion mask give different results?

    # Extracting stats from all images:
    lesions_analysis = analyse_cases(ids=all_ids, labels_dir=TRAIN_LABELS_DIR)

    # Filling NAs:
    cols_w_possible_NAs = [
        'mean_basal_lesion_vol', 'median_basal_lesion_vol', 'mean_new_lesion_vol', 'median_new_lesion_vol'
    ]
    lesions_analysis[cols_w_possible_NAs] = lesions_analysis[cols_w_possible_NAs].fillna(value=0)

    # Stratification:

    # Cutting into categorical:
    lesions_analysis['bl_bin'] = pd.qcut(lesions_analysis['n_basal_lesions'], 3)
    lesions_analysis['nl_bin'] = pd.cut(
        lesions_analysis['n_new_lesions'],
        bins=[0, 1, 5, np.inf],  # TODO Ander 14/4/23: Comment with Eloy whether there is a more suitable split
        right=False
    )
    # Combining both criteria:
    lesions_analysis['stratification_bin'] = (
            lesions_analysis['bl_bin'].astype(str) + ' & ' + lesions_analysis['nl_bin'].astype(str)
    )
    # Unifying continuous rare class for easier splitting:
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
    for train, val in skf.split(X=train_val_ids, y=train_stratification_bins):
        fold = {
            "train": train_val_ids.iloc[train].tolist(),
            "val": train_val_ids.iloc[val].tolist()
        }
        cv_folds.append(fold)
    with open(PREPROCESSED_DATASET_DIR / 'splits_final.json', 'w') as f:
        json.dump(cv_folds, f)
    print("Done!")
