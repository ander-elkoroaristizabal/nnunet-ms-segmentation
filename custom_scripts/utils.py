import os
import pathlib
from typing import List

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label
from tqdm import tqdm

from custom_scripts.A_config import (
    Dataset,
    TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR,
    TEST_IMAGES_DIR, TEST_LABELS_DIR,
    MSSEG2_IMAGES_DIR, MSSEG2_LABELS_DIR, MSSEG2_PREDICTIONS_DIR,
    TERMINATION
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


def get_paths(case_id: str, dataset: Dataset,
              basal_im: bool = False, follow_up_im: bool = False,
              labels: bool = False,
              preds: bool = False):
    ans = list()
    if dataset == Dataset.train_split:
        images_dir = TRAIN_IMAGES_DIR
        labels_dir = TRAIN_LABELS_DIR
        predictions_dir = None
    elif dataset == Dataset.test_split:
        images_dir = TEST_IMAGES_DIR
        labels_dir = TEST_LABELS_DIR
        predictions_dir = TEST_PREDICTIONS_FOLDER
    else:
        images_dir = MSSEG2_IMAGES_DIR
        labels_dir = MSSEG2_LABELS_DIR
        predictions_dir = MSSEG2_PREDICTIONS_DIR

    if basal_im:
        ans.append(os.path.join(images_dir, f"{case_id}_0000" + TERMINATION))
    if follow_up_im:
        ans.append(os.path.join(images_dir, f"{case_id}_0001" + TERMINATION))
    if labels:
        ans.append(os.path.join(labels_dir, case_id + TERMINATION))
    if preds:
        ans.append(os.path.join(predictions_dir, case_id + TERMINATION))
    return tuple(ans)


def read_images(case_id: str, dataset: Dataset):
    b_image_path, fu_image_path = get_paths(case_id=case_id, dataset=dataset,
                                            basal_im=True, follow_up_im=True)
    b_image = nib.load(b_image_path).get_fdata()
    fu_image = nib.load(fu_image_path).get_fdata()
    return b_image, fu_image


def read_labels(case_id: str, dataset: Dataset):
    labels_path = get_paths(case_id=case_id, dataset=dataset, labels=True)[0]
    return nib.load(labels_path).get_fdata()


def get_lesions_locations(labels, lesion_class: int):
    lesions_identified, n_lesions = label((labels == lesion_class).astype(int),
                                          structure=np.ones(shape=(3, 3, 3)))
    lesions2locations = dict()
    for lesion_id in range(1, n_lesions + 1):
        lesion_locations = np.where(lesions_identified == lesion_id)
        lesions2locations[lesion_id] = [
            (x, y, z) for x, y, z
            in zip(lesion_locations[0], lesion_locations[1], lesion_locations[2])
        ]
    return lesions2locations


ConfArrayVal2Type = {
    1: "TP",
    2: "FP",
    3: "FN"
}


def get_lesion_confusion_array(labels, predictions, lesion_class: int, masked: bool):
    confusion_array = np.zeros(labels.shape)
    lesion_predictions = np.copy(predictions)
    lesion_predictions[lesion_predictions != lesion_class] = -1  # Different from 0
    lesion_labels = np.copy(labels)
    lesion_labels[lesion_labels != lesion_class] = 0
    confusion_array[lesion_predictions == lesion_labels] = 1  # TP
    confusion_array[(lesion_predictions == lesion_class) & (lesion_labels != lesion_class)] = 2  # FP
    confusion_array[(lesion_predictions != lesion_class) & (lesion_labels == lesion_class)] = 3  # FN
    if masked:
        confusion_array = np.ma.masked_where(confusion_array == 0, confusion_array)
    return confusion_array
