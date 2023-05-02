import torch

from custom_scripts.A_config import CONFIGURATION
from nnunetv2.run.run_training import run_training

if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    CPU = torch.device('cpu')

    run_training(
        dataset_name_or_id="100",
        fold="2",  # "3"
        device=CPU,
        configuration=CONFIGURATION,
        trainer_class_name='nnUNetTrainerExtremeOversamplingEarlyStopping',
        export_validation_probabilities=False,
        only_run_validation=True
    )
