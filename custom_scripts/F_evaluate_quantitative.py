import os

import torch

from custom_scripts.A_config import (
    TEST_IMAGES_DIR,
    TEST_LABELS_DIR,
    NNUNET_TEST_RESULTS_PATH,
    DATASET,
    PLAN,
    CONFIGURATION
)
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple

if __name__ == '__main__':
    # Basic configuration
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    CPU = torch.device('cpu')
    CUDA = torch.device('cuda')

    # Configuration to be modified
    TRAINER = "nnUNetTrainerExtremeOversamplingEarlyStopping"
    FOLDS = "4"
    CHECKPOINT_TB_USED = "checkpoint_best.pth"
    extra_info = "__FOLD4"

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
        device=CPU
    )

    compute_metrics_on_folder_simple(
        folder_ref=TEST_LABELS_DIR,
        folder_pred=TEST_PREDICTIONS_FOLDER,
        labels=[1, 2]
    )
