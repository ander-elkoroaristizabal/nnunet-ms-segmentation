We just need to change checkpoint_final.pth -> REAL_checkpoint_final.pth, checkpoint_best.pth-> checkpoint_final.pth,
and run

> nnUNetv2_train 100 3d_fullres (fold) -device cuda -tr nnUNetTrainerEarlyStopping --val --npz

I.e., after all name changes:

> nnUNetv2_train 100 3d_fullres 0 -device cuda -tr $trainer --val --npz
> nnUNetv2_train 100 3d_fullres 1 -device cuda -tr $trainer --val --npz
> nnUNetv2_train 100 3d_fullres 2 -device cuda -tr $trainer --val --npz
> nnUNetv2_train 100 3d_fullres 3 -device cuda -tr $trainer --val --npz
> nnUNetv2_train 100 3d_fullres 4 -device cuda -tr $trainer --val --npz