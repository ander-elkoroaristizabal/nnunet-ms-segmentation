# Config:
dataset=100
conf=3d_fullres
trainer=nnUNetTrainerExtremeOversamplingEarlyStoppingLowLR
input_folder=./data/input
output_folder=./data/output

# Predict
nnUNetv2_predict -d $dataset -i $input_folder -o $output_folder -f  0 1 2 3 4 -tr $trainer -c $conf -p nnUNetPlans

# Rename files:
cd $output_folder
for file in *.nii.gz; do mv "$file" "mask_$file"; done

# Delete plans.json file:
rm plans.json