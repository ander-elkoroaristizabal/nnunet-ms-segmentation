import os
from typing import Dict

import pandas as pd
import torch

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from custom_scripts.A_config import (
    TERMINATION,
    TEST_IMAGES_DIR,
    TEST_LABELS_DIR,
    NNUNET_TEST_RESULTS_PATH,
    DATASET,
    PLAN,
    CONFIGURATION
)
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple, load_summary_json


def get_paths(case_id: str, basal_im: bool = False, follow_up_im: bool = False, labels: bool = False,
              preds: bool = False):
    ans = list()
    if basal_im:
        ans.append(os.path.join(TEST_IMAGES_DIR, f"{case_id}_0000" + TERMINATION))
    if follow_up_im:
        ans.append(os.path.join(TEST_IMAGES_DIR, f"{case_id}_0001" + TERMINATION))
    if labels:
        ans.append(os.path.join(TEST_LABELS_DIR, case_id + TERMINATION))
    if preds:
        ans.append(os.path.join(TEST_PREDICTIONS_FOLDER, case_id + TERMINATION))
    return tuple(ans)


def read_images(case_id: str):
    b_image_path, fu_image_path = get_paths(case_id=case_id, basal_im=True, follow_up_im=True)
    b_image = nib.load(b_image_path).get_fdata()
    fu_image = nib.load(fu_image_path).get_fdata()
    return b_image, fu_image


def read_labels_and_preds(case_id: str):
    labels_path, predictions_path = get_paths(case_id=case_id, labels=True, preds=True)
    labels = nib.load(labels_path).get_fdata()
    predictions = nib.load(predictions_path).get_fdata()
    return labels, predictions


def format_results_into_df(all_results: Dict):
    results = []
    for case_metrics in all_results['metric_per_case']:
        case_id = case_metrics['prediction_file'].split("/")[-1][:-7]
        basal_metrics = {f"b_{key}": value for key, value in case_metrics['metrics'][1].items()}
        new_lesions_metrics = {f"n_{key}": value for key, value in case_metrics['metrics'][2].items()}
        results.append({'case_id': case_id} | basal_metrics | new_lesions_metrics)
    return pd.DataFrame.from_records(data=results)


if __name__ == '__main__':
    # Basic configuration
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    CUDA = torch.device('cuda')

    # Configuration to be modified
    TRAINER = "nnUNetTrainerExtremeOversamplingEarlyStoppingLowLR"
    FOLDS = "0 1 2 3 4"
    CHECKPOINT_TB_USED = "checkpoint_final.pth"
    extra_info = ""  # "__FOLD4"

    # Derived settings:
    MODEL_FOLDER_NAME = TRAINER + '__' + PLAN + "__" + CONFIGURATION
    TEST_SPECIFIC_FOLDER_NAME = MODEL_FOLDER_NAME + extra_info
    TEST_PREDICTIONS_FOLDER = os.path.join(NNUNET_TEST_RESULTS_PATH, DATASET, TEST_SPECIFIC_FOLDER_NAME)

    predict_from_raw_data(
        list_of_lists_or_source_folder=TEST_IMAGES_DIR,
        output_folder=TEST_PREDICTIONS_FOLDER,
        model_training_output_dir=MODEL_FOLDER_NAME,
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

    # TODO Ander 2/5/23: Temporal!
    TEST_PREDICTIONS_FOLDER = "/home/ander/PycharmProjects/nnunet-ms-segmentation/" \
                              "nnUNet_test_results/Dataset100_MSSEG/" \
                              "nnUNetTrainerFullOversamplingEarlyStopping__nnUNetPlans__3d_fullres"
    results_dict = load_summary_json(filename=os.path.join(TEST_PREDICTIONS_FOLDER, 'summary.json'))
    results_df = format_results_into_df(results_dict)
    print(results_df.loc[:, (results_df.columns == "case_id") | (results_df.columns.str.startswith("n_"))])

    # Confusion matrices (require prediction files!):
    test_images = os.listdir(TEST_IMAGES_DIR)
    test_ids = sorted({file_name.split(".")[0][:-5] for file_name in test_images})
    all_labels = dict()
    all_preds = dict()
    for case in test_ids:
        sample_labels, sample_predictions = read_labels_and_preds(case_id=case)
        all_labels[case] = sample_labels
        all_preds[case] = sample_predictions
    all_flattened_labels = np.concatenate([labels.flatten() for labels in all_labels.values()])
    all_flattened_preds = np.concatenate([preds.flatten() for preds in all_preds.values()])

    cm = confusion_matrix(y_true=all_flattened_labels, y_pred=all_flattened_preds)
    ConfusionMatrixDisplay(cm)
    plt.show()
