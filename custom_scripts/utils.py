"""
This module contains all the utility functions used by several scripts.
"""
import os
import pathlib
from typing import List, Tuple, Dict

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


def extract_id_from_image_filename(image_file_name: str) -> str:
    """Convert an image file name in nnU-Net format to its (nnU-Net) identifier."""
    return image_file_name.split(".")[0][:-5]


def detect_lesions(mask: nib.nifti1.Nifti1Image) -> Tuple[np.ndarray, Dict[str, List[int]]]:
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
    """Analyse cases with id in 'ids' and whose labels are in directory 'labels_dir'.

    Args:
        ids: list of ids of the cases to be analysed
        labels_dir: directory where the labels of the ids are stored

    Returns:
        DataFrame with all results.
    """
    # We iterate over ids and gather all results in a list:
    analysis_results = []
    for case_id in tqdm(ids):
        # We load the labels:
        case_mask = nib.load(labels_dir / (case_id + TERMINATION))
        # We detect the lesions (both basal and new ones):
        lesions_map, lesions = detect_lesions(case_mask)
        # We compute the total, mean and median lesion size for both types of lesions:
        total_basal_lesion_vol = (case_mask.get_fdata() == 1).sum()
        total_new_lesion_vol = (case_mask.get_fdata() == 2).sum()
        mean_basal_lesion_vol = np.median([(lesions_map == b_lesion).sum() for b_lesion in lesions['basal']])
        median_basal_lesion_vol = np.mean([(lesions_map == b_lesion).sum() for b_lesion in lesions['basal']])
        mean_new_lesion_vol = np.median([(lesions_map == b_lesion).sum() for b_lesion in lesions['new']])
        median_new_lesion_vol = np.mean([(lesions_map == b_lesion).sum() for b_lesion in lesions['new']])
        # And we append the results to the list, including the number of lesions:
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
    # Finally, we return all results as a dataframe:
    return pd.DataFrame.from_records(analysis_results)


def get_paths(case_id: str, dataset: Dataset,
              basal_im: bool = False, follow_up_im: bool = False,
              labels: bool = False,
              preds: bool = False):
    """Get the path(s) to the basal image, follow-up image, labels and/or preds.

    Args:
        case_id: identifier of the case
        dataset: dataset the case belongs to
        basal_im: whether to return the path to the basal image
        follow_up_im: whether to return the path to the follow-up image
        labels: whether to return the path to the labels
        preds: whether to return the path to the preds

    Returns:
        Paths to the requested files in the order of the arguments.
    """
    # We get the images, labels and predictions paths using the dataset:
    if dataset == Dataset.train_split:
        images_dir = TRAIN_IMAGES_DIR
        labels_dir = TRAIN_LABELS_DIR
        predictions_dir = None
    elif dataset == Dataset.test_split:
        images_dir = TEST_IMAGES_DIR
        labels_dir = TEST_LABELS_DIR
        # TEST_PREDICTIONS_FOLDER will be defined externally before being used
        predictions_dir = TEST_PREDICTIONS_FOLDER
    else:
        images_dir = MSSEG2_IMAGES_DIR
        labels_dir = MSSEG2_LABELS_DIR
        predictions_dir = MSSEG2_PREDICTIONS_DIR

    # We gather requested paths:
    ans = list()
    if basal_im:
        ans.append(os.path.join(images_dir, f"{case_id}_0000" + TERMINATION))
    if follow_up_im:
        ans.append(os.path.join(images_dir, f"{case_id}_0001" + TERMINATION))
    if labels:
        ans.append(os.path.join(labels_dir, case_id + TERMINATION))
    if preds:
        ans.append(os.path.join(predictions_dir, case_id + TERMINATION))
    # And return them:
    return tuple(ans)


def read_images(case_id: str, dataset: Dataset):
    """Read the images corresponding to the case.

    Args:
        case_id: identified of the case
        dataset: dataset to which the case belongs

    Returns:
        The arrays of the basal and follow-up images
    """
    b_image_path, fu_image_path = get_paths(case_id=case_id, dataset=dataset,
                                            basal_im=True, follow_up_im=True)
    b_image = nib.load(b_image_path).get_fdata()
    fu_image = nib.load(fu_image_path).get_fdata()
    return b_image, fu_image


def read_labels(case_id: str, dataset: Dataset):
    """Read the labels corresponding to the case.

        Args:
            case_id: identified of the case
            dataset: dataset to which the case belongs

        Returns:
            The array with the labels.
        """
    labels_path = get_paths(case_id=case_id, dataset=dataset, labels=True)[0]
    return nib.load(labels_path).get_fdata()


def get_lesions_locations(labels: np.ndarray, lesion_class: int):
    """Get the location of all lesions of class 'lesion_class'.

    Args:
        labels: array with the labels
        lesion_class: class of the lesions to be identified

    Returns:
        A dictionary with lesions as keys and their voxels as values.
    """
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


# Dictionary that maps values in confusion arrays
# (defined below) to their meaning:
ConfArrayVal2Type = {
    1: "TP",
    2: "FP",
    3: "FN"
}


def get_lesion_confusion_array(labels: np.ndarray, predictions: np.ndarray, lesion_class: int, masked: bool):
    """Get the confusions the model makes when detecting the 'lesion_class'.

    If 'masked' masks irrelevant values, better for plotting.

    Args:
        labels: labels array
        predictions: predictions array
        lesion_class: lesion on which we want to evaluate
        masked: whether to mask or not

    Returns:
        Array with the confusions made by the model.
    """
    # We initialize the confusions array:
    confusion_array = np.zeros(labels.shape)
    # Make a copy of the predictions and make those belonging to another class:
    lesion_predictions = np.copy(predictions)
    lesion_predictions[lesion_predictions != lesion_class] = -1  # Different from 0
    # Make a copy of the labels and mark labels belonging to another class:
    lesion_labels = np.copy(labels)
    lesion_labels[lesion_labels != lesion_class] = 0  # Different from -1
    # Get TPs, FPs and FNs:
    # TPs are now all cases where lesions == prediction:
    confusion_array[lesion_predictions == lesion_labels] = 1
    # And FPs and FNs are easy to get:
    confusion_array[(lesion_predictions == lesion_class) & (lesion_labels != lesion_class)] = 2  # FP
    confusion_array[(lesion_predictions != lesion_class) & (lesion_labels == lesion_class)] = 3  # FN
    # If masked, mask values:
    if masked:
        confusion_array = np.ma.masked_where(confusion_array == 0, confusion_array)
    return confusion_array
