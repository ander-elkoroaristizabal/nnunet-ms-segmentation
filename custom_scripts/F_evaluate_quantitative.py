"""
This script performs the quantitative analysis of the performance of our ensemble of models.
"""
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import label
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

import custom_scripts.utils as utils
from custom_scripts.A_config import (
    TEST_IMAGES_DIR,
    NNUNET_TEST_RESULTS_PATH,
    DATASET,
    PLAN,
    CONFIGURATION,
    MSSEG2_PREDICTIONS_DIR,
    MSSEG2_LABELS_DIR,
    NNUNET_RESULTS_PATH,
    TEST_LABELS_DIR,
    Dataset
)
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple, load_summary_json
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data


def read_labels_and_preds(case_id: str, dataset: Dataset):
    """Get the labels and predictions arrays for a given case in a dataset.

    Args:
        case_id: identifier of the case
        dataset: dataset to which it belongs

    Returns:
        labels and predictions arrays
    """
    # Update with the directory where the test predictions are:
    if dataset == Dataset.test_split:
        utils.TEST_PREDICTIONS_FOLDER = TEST_PREDICTIONS_FOLDER
    # Get paths:
    labels_path, predictions_path = utils.get_paths(
        case_id=case_id, dataset=dataset,
        labels=True, preds=True
    )
    # Load files and return them:
    labels = nib.load(labels_path).get_fdata()
    predictions = nib.load(predictions_path).get_fdata()
    return labels, predictions


def format_results_into_df(all_results: Dict):
    """Format a results dict as outputted by nnU-Net into a dataframe.

    Args:
        all_results: dictionary of results outputted by nnU-Net.

    Returns:
        DataFrame with results.
    """
    results = []
    for case_metrics in all_results['metric_per_case']:
        case_id = case_metrics['prediction_file'].split("/")[-1][:-7]
        # Extract and rename basal lesion metrics:
        if 1 in case_metrics['metrics'].keys():
            basal_metrics = {f"b_{key}": value for key, value in case_metrics['metrics'][1].items()}
        else:
            basal_metrics = dict()
        # Extract and rename new or evolving lesion metrics:
        if 2 in case_metrics['metrics'].keys():
            new_lesions_metrics = {f"new_{key}": value for key, value in case_metrics['metrics'][2].items()}
        else:
            new_lesions_metrics = {}
        # Store results:
        results.append({'case_id': case_id} | basal_metrics | new_lesions_metrics)
    # Return dataframe with all results:
    return pd.DataFrame.from_records(data=results)


def get_all_preds_and_labels(ids: List[str], dataset: Dataset):
    """Get the labels and predictions dicts for all cases in 'ids'.

    Args:
        ids: cases whose labels and preds need to be extracted
        dataset: dataset to which the cases belong

    Returns:
        Dict with id -> labels, Dict with id -> predictions
    """
    labels = dict()
    preds = dict()
    for case in tqdm(ids):
        sample_labels, sample_predictions = read_labels_and_preds(case_id=case, dataset=dataset)
        labels[case] = sample_labels
        preds[case] = sample_predictions
    return labels, preds


def compute_lesion_level_metrics(labels: np.ndarray, preds: np.ndarray, lesion_class: int):
    """Computes lesion level metrics for lesion class 'lesion_class'.

    Args:
        labels: array with labels
        preds: array with predictions
        lesion_class: class of the lesions to be analysed

    Returns:
        Number of true lesions, number of predicted lesions, TPs, FPs, FNs
    """
    # TODO Ander 11/6/23: Refactor
    # We get the array with the true different lesions of the lesion class and their number:
    gt_lesions, n_gt_lesions = label((labels == lesion_class).astype(int))
    # Removing lesions smaller than 3mm:
    for lesion_id in range(1, n_gt_lesions + 1):
        lesion_mask = gt_lesions == lesion_id
        lesion_size = lesion_mask.sum()
        if lesion_size <= 3:
            gt_lesions[lesion_mask] = 0
    # We get the array with the predicted different lesions of the lesion class and their number:
    predicted_lesions, n_predicted_lesions = label((preds == lesion_class).astype(int))
    for lesion_id in range(1, n_predicted_lesions + 1):
        lesion_mask = predicted_lesions == lesion_id
        lesion_size = lesion_mask.sum()
        if lesion_size <= 3:
            predicted_lesions[lesion_mask] = 0
    # Making lesion names disjoint:
    new_predicted_lesions = np.copy(predicted_lesions)
    for pred_lesion_id in range(1, n_predicted_lesions + 1):
        new_pred_lesion_id = pred_lesion_id + n_gt_lesions
        new_predicted_lesions[predicted_lesions == pred_lesion_id] = new_pred_lesion_id
    # We compute TPs, FPs and FNs taking into account that
    # two predicted lesions may correspond to a single real or vice-versa
    # and requiring that 10% of the voxels are covered:
    # TP_G (detected ground truth lesions):
    tp_gt = set()
    for gt_lesion in range(1, n_gt_lesions + 1):
        lesion_mask = gt_lesions == gt_lesion
        overlapping_preds = new_predicted_lesions[lesion_mask] > 0
        if overlapping_preds.sum() > (0.1 * lesion_mask.sum()):
            tp_gt |= {gt_lesion}
    # TP_A (predicted lesions covered by GT lesions):
    tp_pred = set()
    for pred_lesion in range(n_gt_lesions + 1, n_predicted_lesions + n_gt_lesions + 1):
        lesion_mask = new_predicted_lesions == pred_lesion
        overlapping_gts = gt_lesions[lesion_mask] > 0
        if overlapping_gts.sum() > (0.1 * lesion_mask.sum()):
            tp_pred |= {pred_lesion}
    fp = set(np.unique(new_predicted_lesions[(gt_lesions == 0) & (new_predicted_lesions > 0)])) - tp_pred
    fn = set(np.unique(gt_lesions[(gt_lesions > 0) & (new_predicted_lesions == 0)])) - tp_gt
    tp = len(tp_gt)
    fp = len(fp)
    fn = len(fn)
    # We make sure that our computations make sense:
    n_predicted_lesions = len(np.unique(new_predicted_lesions)) - 1
    n_gt_lesions = len(np.unique(gt_lesions)) - 1
    assert len(tp_pred) + fp == n_predicted_lesions, f"{len(tp_pred) = }, {fp = }, {n_predicted_lesions = }"
    assert len(tp_gt) + fn == n_gt_lesions, f"{len(tp_gt) = }, {fn = }, {n_gt_lesions = }"
    # Compute the F1-Score:
    if n_gt_lesions == 0:
        f1 = np.NAN
    elif (n_gt_lesions > 0) & (n_predicted_lesions == 0):
        f1 = 0
    else:
        se_l = len(tp_gt) / n_gt_lesions
        p_l = len(tp_pred) / n_predicted_lesions
        try:
            f1 = 2 * se_l * p_l / (se_l + p_l)
        except ZeroDivisionError:
            f1 = 0
    # And return the results:
    return n_gt_lesions, n_predicted_lesions, tp, fp, fn, f1


def compute_all_lesion_level_metrics(
        ids: List[str],
        labels_dict: Dict[str, np.ndarray],
        preds_dict: Dict[str, np.ndarray],
        lesion_class: int = 2) -> pd.DataFrame:
    """Compute the lesion-level metrics for cases in 'ids' and lesions of class 'lesion_class'.

    Args:
        ids: ids of the cases to be analysed
        labels_dict: dictionary with the labels of the cases
        preds_dict: dictionary with the predictions of the cases
        lesion_class: lesion class to be used

    Returns:
        DataFrame with lesion-level metrics.
    """
    lesion_level_metrics = []
    lesion_str = "new" if lesion_class == 2 else "basal"
    for case in tqdm(ids):
        # We get lesion-level metrics:
        n_gt, n_pred, tp, fp, fn, f1 = compute_lesion_level_metrics(labels=labels_dict[case],
                                                                    preds=preds_dict[case],
                                                                    lesion_class=lesion_class)
        # Put together all metrics:
        lesion_level_metrics.append({
            "case_id": case,
            f"n_ref_{lesion_str}_lesions": n_gt,
            f"n_pred_{lesion_str}_lesions": n_pred,
            f"{lesion_str}_lesion_tp": tp,
            f"{lesion_str}_lesion_fp": fp,
            f"{lesion_str}_lesion_fn": fn,
            f"{lesion_str}_lesion_F1": f1
        })
    # And return a dataframe with all results.
    return pd.DataFrame.from_records(data=lesion_level_metrics)


if __name__ == '__main__':
    # Basic configuration
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    CUDA = torch.device('cuda')

    # Configuration to be modified
    PREDICT = False  # Change to true if you need to predict
    TRAINER = "nnUNetTrainerExtremeOversamplingEarlyStoppingLowLR"
    FOLDS = (0, 1, 2, 3, 4)
    CHECKPOINT_TB_USED = "checkpoint_final.pth"
    extra_info = ""

    # Derived settings:
    MODEL_FOLDER_NAME = TRAINER + '__' + PLAN + "__" + CONFIGURATION
    TEST_SPECIFIC_FOLDER_NAME = MODEL_FOLDER_NAME + extra_info
    TEST_PREDICTIONS_FOLDER = os.path.join(NNUNET_TEST_RESULTS_PATH, DATASET, TEST_SPECIFIC_FOLDER_NAME)

    if PREDICT:
        predict_from_raw_data(
            list_of_lists_or_source_folder=str(TEST_IMAGES_DIR),
            output_folder=TEST_PREDICTIONS_FOLDER,
            model_training_output_dir=str(NNUNET_RESULTS_PATH / DATASET / MODEL_FOLDER_NAME),
            use_folds=FOLDS,
            verbose=True,
            checkpoint_name=CHECKPOINT_TB_USED,
            device=CUDA
        )

        compute_metrics_on_folder_simple(
            folder_ref=TEST_LABELS_DIR,
            folder_pred=TEST_PREDICTIONS_FOLDER,
            labels=[1, 2]
        )

    # Loading results:

    # TEST:
    test_results_dict = load_summary_json(TEST_PREDICTIONS_FOLDER + "/" + 'summary.json')
    test_results_df = format_results_into_df(test_results_dict)

    # Getting labels and predictions:
    test_images = utils.list_all_nifti_files(TEST_IMAGES_DIR)
    test_ids = sorted({utils.extract_id_from_image_filename(file_name) for file_name in test_images})
    all_labels, all_preds = get_all_preds_and_labels(ids=test_ids, dataset=Dataset.test_split)

    # Adding lesion-wise results:
    b_test_lesion_wise_res = compute_all_lesion_level_metrics(ids=test_ids, labels_dict=all_labels,
                                                              preds_dict=all_preds, lesion_class=1)
    n_test_lesion_wise_res = compute_all_lesion_level_metrics(ids=test_ids, labels_dict=all_labels,
                                                              preds_dict=all_preds)

    test_results_df = test_results_df.merge(n_test_lesion_wise_res, on='case_id')
    test_results_df = test_results_df.merge(b_test_lesion_wise_res, on='case_id')

    # Column subsets:
    new_cols_mask = (test_results_df.columns == "case_id") | test_results_df.columns.str.contains("new_")
    basal_cols_mask = ((test_results_df.columns == "case_id")
                       | test_results_df.columns.str.contains("basal_")
                       | test_results_df.columns.str.contains("b_"))
    # Basal lesion results:
    print(test_results_df.loc[:, basal_cols_mask].describe())
    # New lesion results:
    # Results on no-lesions subset:
    print(test_results_df.loc[test_results_df.new_n_ref == 0, new_cols_mask].describe())
    # Results on with-lesions subset:
    print(test_results_df.loc[test_results_df.new_n_ref > 0, new_cols_mask].describe())
    del all_preds, all_labels  # Memory issues

    # MSSEG2:
    MSSEG2_results_dict = load_summary_json(MSSEG2_PREDICTIONS_DIR / 'summary.json')
    MSSEG2_results_df = format_results_into_df(MSSEG2_results_dict)

    # Getting labels and predictions:
    test_images = utils.list_all_nifti_files(MSSEG2_LABELS_DIR)
    test_ids = sorted({file_name[:-7] for file_name in test_images})
    all_labels, all_preds = get_all_preds_and_labels(ids=test_ids, dataset=Dataset.msseg2)

    # Adding lesion-wise results:
    lesion_wise_res = compute_all_lesion_level_metrics(ids=test_ids, labels_dict=all_labels, preds_dict=all_preds)
    MSSEG2_results_df = MSSEG2_results_df.merge(lesion_wise_res, on='case_id')
    M_new_cols_mask = (MSSEG2_results_df.columns == "case_id") | MSSEG2_results_df.columns.str.contains("new_")

    # Results by subset:
    # Bad resolution images (from MS Open Data):
    # All cases have new lesions:
    print(MSSEG2_results_df.loc[MSSEG2_results_df.case_id.str.contains('patient'), M_new_cols_mask].describe())
    # Good resolution images (from MSSEG-2):
    msseg2_cases = ~MSSEG2_results_df.case_id.str.contains('patient')
    # Results on no-lesions subset:
    print(MSSEG2_results_df.loc[msseg2_cases & (MSSEG2_results_df.new_n_ref == 0), M_new_cols_mask].describe())
    # Results on with-lesions subset:
    print(MSSEG2_results_df.loc[msseg2_cases & (MSSEG2_results_df.new_n_ref > 0), M_new_cols_mask].describe())

    # Confusion matrix:
    all_flattened_labels = np.concatenate([labels.flatten() for labels in all_labels.values()])
    all_flattened_preds = np.concatenate([preds.flatten() for preds in all_preds.values()])
    cm = confusion_matrix(y_true=all_flattened_labels, y_pred=all_flattened_preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig("TestConfusionMatrix.png")
    plt.show()
