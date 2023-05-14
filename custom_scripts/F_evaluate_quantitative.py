import os
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

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
    CONFIGURATION,
    MSSEG2_PREDICTIONS_DIR,
    MSSEG2_LABELS_DIR,
    MSSEG2_IMAGES_DIR,
    TestDataset
)
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple, load_summary_json
from scipy.ndimage import label


def get_paths(case_id: str, test_dataset: TestDataset, basal_im: bool = False, follow_up_im: bool = False,
              labels: bool = False,
              preds: bool = False):
    ans = list()
    if test_dataset == TestDataset.msseg2:
        images_dir = MSSEG2_IMAGES_DIR
        labels_dir = MSSEG2_LABELS_DIR
        predictions_dir = MSSEG2_PREDICTIONS_DIR
    else:
        images_dir = TEST_IMAGES_DIR
        labels_dir = TEST_LABELS_DIR
        predictions_dir = TEST_PREDICTIONS_FOLDER
    if basal_im:
        ans.append(os.path.join(images_dir, f"{case_id}_0000" + TERMINATION))
    if follow_up_im:
        ans.append(os.path.join(images_dir, f"{case_id}_0001" + TERMINATION))
    if labels:
        ans.append(os.path.join(labels_dir, case_id + TERMINATION))
    if preds:
        ans.append(os.path.join(predictions_dir, case_id + TERMINATION))
    return tuple(ans)


def read_images(case_id: str, test_dataset: TestDataset):
    b_image_path, fu_image_path = get_paths(case_id=case_id, test_dataset=test_dataset,
                                            basal_im=True, follow_up_im=True)
    b_image = nib.load(b_image_path).get_fdata()
    fu_image = nib.load(fu_image_path).get_fdata()
    return b_image, fu_image


def read_labels_and_preds(case_id: str, test_dataset: TestDataset):
    labels_path, predictions_path = get_paths(case_id=case_id, test_dataset=test_dataset,
                                              labels=True, preds=True)
    labels = nib.load(labels_path).get_fdata()
    predictions = nib.load(predictions_path).get_fdata()
    return labels, predictions


def format_results_into_df(all_results: Dict):
    results = []
    for case_metrics in all_results['metric_per_case']:
        case_id = case_metrics['prediction_file'].split("/")[-1][:-7]
        if 1 in case_metrics['metrics'].keys():
            basal_metrics = {f"b_{key}": value for key, value in case_metrics['metrics'][1].items()}
        else:
            basal_metrics = dict()
        if 2 in case_metrics['metrics'].keys():
            new_lesions_metrics = {f"new_{key}": value for key, value in case_metrics['metrics'][2].items()}
        else:
            new_lesions_metrics = {}
        results.append({'case_id': case_id} | basal_metrics | new_lesions_metrics)
    return pd.DataFrame.from_records(data=results)


def get_all_preds_and_labels(ids: List[str], test_dataset: TestDataset):
    labels = dict()
    preds = dict()
    for case in tqdm(ids):
        sample_labels, sample_predictions = read_labels_and_preds(case_id=case, test_dataset=test_dataset)
        labels[case] = sample_labels
        preds[case] = sample_predictions
    return labels, preds


def compute_lesion_level_metrics(labels, preds, lesion_class: int):
    gt_lesions, n_gt_lesions = label((labels == lesion_class).astype(int),
                                     structure=np.ones(shape=(3, 3, 3)))
    predicted_lesions, n_predicted_lesions = label((preds == lesion_class).astype(int),
                                                   structure=np.ones(shape=(3, 3, 3)))
    # Making lesion names disjoint:
    new_predicted_lesions = np.copy(predicted_lesions)
    for pred_lesion_id in range(1, n_predicted_lesions + 1):
        # print(f"{pred_lesion_id = }")
        new_pred_lesion_id = pred_lesion_id + n_gt_lesions
        # print(f"{new_pred_lesion_id = }")
        new_predicted_lesions[predicted_lesions == pred_lesion_id] = new_pred_lesion_id
    tp_gt = set(np.unique(gt_lesions[(gt_lesions > 0) & (predicted_lesions > 0)]))
    tp_pred = set(np.unique(new_predicted_lesions[(gt_lesions > 0) & (new_predicted_lesions > 0)]))
    fp = set(np.unique(new_predicted_lesions[(gt_lesions == 0) & (new_predicted_lesions > 0)])) - tp_pred
    fn = set(np.unique(gt_lesions[(gt_lesions > 0) & (new_predicted_lesions == 0)])) - tp_gt
    tn = set(np.unique(gt_lesions[(gt_lesions == 0) & (new_predicted_lesions == 0)]))
    tp = len(tp_gt)
    fp = len(fp)
    fn = len(fn)
    assert len(tp_pred) + fp == n_predicted_lesions, f"{tp_pred = }, {fp = }, {n_predicted_lesions = }"
    assert len(tp_gt) + fn == n_gt_lesions, f"{tp_gt = }, {fn = }, {n_gt_lesions = }"
    return n_gt_lesions, n_predicted_lesions, tp, fp, fn


def compute_all_lesion_level_metrics(ids, labels_dict, preds_dict, lesion_class=2):
    lesion_level_metrics = []
    lesion_str = "new" if lesion_class == 2 else "basal"
    for case in tqdm(ids):
        n_gt, n_pred, tp, fp, fn = compute_lesion_level_metrics(labels=labels_dict[case],
                                                                preds=preds_dict[case],
                                                                lesion_class=lesion_class)
        try:
            F1 = 2 * tp / (2 * tp + fp + fn)
        except ZeroDivisionError:
            F1 = np.NAN
        lesion_level_metrics.append({
            "case_id": case,
            f"n_ref_{lesion_str}_lesions": n_gt,
            f"n_pred_{lesion_str}_lesions": n_pred,
            f"{lesion_str}_lesion_tp": tp,
            f"{lesion_str}_lesion_fp": fp,
            f"{lesion_str}_lesion_fn": fn,
            f"{lesion_str}_lesion_F1": F1
        })
    return pd.DataFrame.from_records(data=lesion_level_metrics)


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
        folder_ref=MSSEG2_LABELS_DIR,
        folder_pred=MSSEG2_PREDICTIONS_DIR,
        labels=[2]
    )

    # Loading results:

    # TEST:
    test_results_dict = load_summary_json(TEST_PREDICTIONS_FOLDER + "/" + 'summary.json')
    test_results_df = format_results_into_df(test_results_dict)

    # Getting labels and predictions:
    test_images = os.listdir(TEST_IMAGES_DIR)
    test_ids = sorted({file_name.split(".")[0][:-5] for file_name in test_images})
    all_labels, all_preds = get_all_preds_and_labels(ids=test_ids, test_dataset=TestDataset.test_split)

    # Adding lesion-wise results:
    test_lesion_wise_res = compute_all_lesion_level_metrics(ids=test_ids, labels_dict=all_labels, preds_dict=all_preds)
    test_results_df = test_results_df.merge(test_lesion_wise_res, on='case_id')

    # Results on no-lesions subset:
    new_cols_mask = (test_results_df.columns == "case_id") | test_results_df.columns.str.contains("new_")
    print(test_results_df.loc[test_results_df.new_n_ref == 0, new_cols_mask])
    # Results on with-lesions subset:
    print(test_results_df.loc[test_results_df.new_n_ref > 0, new_cols_mask])
    del all_preds, all_labels  # Memory issues

    # MSSEG2:
    MSSEG2_results_dict = load_summary_json(MSSEG2_PREDICTIONS_DIR / 'summary.json')
    MSSEG2_results_df = format_results_into_df(MSSEG2_results_dict)

    # Getting labels and predictions:
    test_images = os.listdir(MSSEG2_LABELS_DIR)
    test_ids = sorted({file_name[:-7] for file_name in test_images})
    all_labels, all_preds = get_all_preds_and_labels(ids=test_ids, test_dataset=TestDataset.msseg2)

    # Adding lesion-wise results:
    lesion_wise_res = compute_all_lesion_level_metrics(ids=test_ids, labels_dict=all_labels, preds_dict=all_preds)
    MSSEG2_results_df = MSSEG2_results_df.merge(lesion_wise_res, on='case_id')

    # Results on no-lesions subset:
    print(MSSEG2_results_df.loc[MSSEG2_results_df.new_n_ref == 0, :])
    # Results on with-lesions subset:
    print(MSSEG2_results_df.loc[MSSEG2_results_df.new_n_ref > 0, :])

    # Results by subset:
    # Bad resolution images (from MS Open Data):
    print(MSSEG2_results_df[MSSEG2_results_df.case_id.str.contains('patient')].describe())
    # Good resolution images (from MSSEG-2):
    print(MSSEG2_results_df[~MSSEG2_results_df.case_id.str.contains('patient')].describe())

    # Confusion matrix:
    all_flattened_labels = np.concatenate([labels.flatten() for labels in all_labels.values()])
    all_flattened_preds = np.concatenate([preds.flatten() for preds in all_preds.values()])
    cm = confusion_matrix(y_true=all_flattened_labels, y_pred=all_flattened_preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig("TestConfusionMatrix.png")
    plt.show()
