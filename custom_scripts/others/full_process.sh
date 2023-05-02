# Paths:
export nnUNet_raw="./data/nnUNet_raw_data"
export nnUNet_preprocessed="./data/nnUNet_preprocessed_data"
export nnUNet_results="./nnUNet_results"

# Preprocess dataset
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity

# Train folds (we could also use "all" to train all splits):
conf=3d_fullres
trainer=nnUNetTrainerExtremeOversamplingEarlyStoppingLowLR
nnUNetv2_train 100 $conf 0 -device cuda -tr $trainer --npz
nnUNetv2_train 100 $conf 1 -device cuda -tr $trainer --npz
nnUNetv2_train 100 $conf 2 -device cuda -tr $trainer --npz
nnUNetv2_train 100 $conf 3 -device cuda -tr $trainer --npz
nnUNetv2_train 100 $conf 4 -device cuda -tr $trainer --npz

# Find best configuration:
nnUNetv2_find_best_configuration 100 -c $conf -tr $trainer