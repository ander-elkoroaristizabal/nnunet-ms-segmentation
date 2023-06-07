"""
This script performs a quantitative analysis of the dataset,
extracting statistics at both the voxel and the lesion level.
"""
import os

import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import load_json
from matplotlib import pyplot as plt

# Relevant paths:
from custom_scripts.A_config import (
    TRAIN_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    TEST_IMAGES_DIR,
    TEST_LABELS_DIR,
    PREPROCESSED_DATASET_DIR,
    MSSEG2_LABELS_DIR,
    MSSEG2_IMAGES_DIR,
    Dataset
)
from custom_scripts.utils import analyse_cases, extract_id_from_image_filename


def find_validation_fold(case_id: str, folds_dict):
    """Find which validation fold a training case is in."""
    for fold_id, fold_dict in enumerate(folds_dict):
        if case_id in fold_dict['val']:
            return fold_id


if __name__ == "__main__":
    # Whether to use the test split or the MSSEG-2 as test dataset.
    test_dataset = Dataset.test_split

    # And adapting paths to images and labels:
    if test_dataset == Dataset.msseg2:
        TEST_IMAGES_DIR = MSSEG2_IMAGES_DIR
        TEST_LABELS_DIR = MSSEG2_LABELS_DIR

    # Beginning of the analysis:

    # We get both train and test ids:
    train_images = os.listdir(TRAIN_IMAGES_DIR)
    train_ids = sorted({extract_id_from_image_filename(file_name) for file_name in train_images})

    test_images = os.listdir(TEST_IMAGES_DIR)
    test_ids = sorted({extract_id_from_image_filename(file_name) for file_name in test_images})

    # We load the file with the Cross-Validation splits:
    splits_file = os.path.join(PREPROCESSED_DATASET_DIR, "splits_final.json")
    splits = load_json(splits_file)

    # Extracting stats from all images:
    train_analysis_results = analyse_cases(ids=train_ids, labels_dir=TRAIN_LABELS_DIR)
    test_analysis_results = analyse_cases(ids=test_ids, labels_dir=TEST_LABELS_DIR)

    # Extracting those columns where group-by averages won't work well due to granularity:
    train_non_averageable_cols = train_analysis_results.columns[
        (train_analysis_results.columns.str.contains("mean")
         | train_analysis_results.columns.str.contains("median"))
    ].to_list()
    test_non_averageable_cols = test_analysis_results.columns[
        (test_analysis_results.columns.str.contains("mean")
         | test_analysis_results.columns.str.contains("median"))
    ].to_list()

    # And summarizing the stats without these columns:
    train_analysis_results.drop(columns=train_non_averageable_cols).describe()
    test_analysis_results.drop(columns=test_non_averageable_cols).describe()

    # Adding folds information and summarizing by fold:
    train_analysis_results['val_fold'] = train_analysis_results['case_id'].apply(
        find_validation_fold,
        folds_dict=splits)
    fold_wise_results = train_analysis_results.drop(columns=['case_id'] + train_non_averageable_cols).groupby(
        'val_fold').mean()

    # Adding correctly computed mean values to the columns dropped before:
    basal_sums = train_analysis_results[['val_fold', 'total_basal_lesion_vol', 'n_basal_lesions']].groupby(
        'val_fold').sum()
    fold_wise_results['mean_basal_lesion_vol'] = basal_sums['total_basal_lesion_vol'] / basal_sums['n_basal_lesions']
    new_sums = train_analysis_results[['val_fold', 'total_new_lesion_vol', 'n_new_lesions']].groupby(
        'val_fold').sum()
    fold_wise_results['mean_new_lesion_vol'] = new_sums['total_new_lesion_vol'] / new_sums['n_new_lesions']

    # Some plots:
    df = pd.concat([train_analysis_results, test_analysis_results], axis=0)

    # Distribution of new lesions, with and without zero-cases:
    train_analysis_results.n_new_lesions.hist()
    plt.show()
    df.dropna()[['n_new_lesions']].boxplot()
    plt.show()

    # Studying relation between number of basal and evolving lesions:
    plt.scatter(df['n_basal_lesions'], df['n_new_lesions'], s=5)
    plt.title("Basal lesion number vs.\n New or Evolving lesion number")
    plt.xlabel("Number of basal lesions")
    plt.ylabel("Number of new lesions")
    plt.savefig("N_Basal_vs_N_New.png")
    from scipy.stats import pearsonr

    correlation = pearsonr(df['n_basal_lesions'], df['n_new_lesions']).statistic
    print(f"{correlation = }")
