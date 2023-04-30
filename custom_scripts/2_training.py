import torch.onnx
import os

os.environ['nnUNet_raw'] = "./data/nnUNet_raw_data"
os.environ['nnUNet_preprocessed'] = "./data/nnUNet_preprocessed_data"
os.environ['nnUNet_results'] = "./nnUNet_results"

from nnunetv2.run.run_training import run_training

if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device('cpu')

    run_training(
        dataset_name_or_id="100",
        fold="0",
        device=device,
        configuration='3d_fullres',
        trainer_class_name='nnUNetTrainerFullOversamplingEarlyStopping'
    )
