import os
from batchgenerators.utilities.file_and_folder_operations import load_json

# Relevant paths:
from custom_scripts.A_config import (
    TRAIN_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    TEST_IMAGES_DIR,
    TEST_LABELS_DIR,
    PREPROCESSED_DATASET_DIR
)
from custom_scripts.C_analyse_and_split import analyse_cases

# Splits (re-) analysis:

# Ids:

train_images = os.listdir(TRAIN_IMAGES_DIR)
train_ids = sorted({file_name.split(".")[0][:-5] for file_name in train_images})

test_images = os.listdir(TEST_IMAGES_DIR)
test_ids = sorted({file_name.split(".")[0][:-5] for file_name in test_images})

# Splits:
splits_file = os.path.join(PREPROCESSED_DATASET_DIR, "splits_final.json")
splits = load_json(splits_file)

# Extracting stats from all images:
train_analysis_results = analyse_cases(ids=train_ids, labels_dir=TRAIN_LABELS_DIR)
test_analysis_results = analyse_cases(ids=test_ids, labels_dir=TEST_LABELS_DIR)

train_analysis_results.drop(columns='case_id').mean()
test_analysis_results.drop(columns='case_id').mean()


# Adding folds information

def find_validation_fold(case_id: str, folds=splits):
    for fold_id, fold_dict in enumerate(folds):
        if case_id in fold_dict['val']:
            return fold_id


train_analysis_results['val_fold'] = train_analysis_results['case_id'].apply(find_validation_fold, folds=splits)
fold_wise_results = train_analysis_results.drop(columns='case_id').groupby('val_fold').mean()
