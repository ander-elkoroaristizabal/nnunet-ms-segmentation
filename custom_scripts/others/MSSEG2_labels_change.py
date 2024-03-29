"""
This script performs the conversion of the MSSEG-2 + MS Open Data datasets to a format suitable for nnU-Net.
"""
import os
import re

import nibabel

from custom_scripts.A_config import MSSEG2_IMAGES_DIR, MSSEG2_LABELS_DIR
from custom_scripts.utils import list_all_nifti_files

if __name__ == "__main__":
    # Renaming image files:
    image_files = list_all_nifti_files(MSSEG2_IMAGES_DIR)
    for file in image_files:
        prefix, time_str, suffix = re.findall(r"^(.*?)_time(\d+)_FL(\.nii\.gz)$", file)[0]
        new_file = f"{prefix}_{str(int(time_str) - 1).zfill(4)}{suffix}"
        os.rename(file, new_file)

    # Renaming of labels:
    label_files = list_all_nifti_files(MSSEG2_LABELS_DIR)
    for file in label_files:
        new_file = re.sub(r"mask", "", file)
        os.rename(file, new_file)

    # Changing labels from 1 to 2 in order to fit our predictions:
    label_files = list_all_nifti_files(MSSEG2_LABELS_DIR)
    for label_file_name in label_files:
        label_file = nibabel.load(MSSEG2_LABELS_DIR / label_file_name)
        seg = label_file.get_fdata()
        seg[seg == 1] = 2
        new_seg = nibabel.Nifti1Image(seg, label_file.affine, label_file.header)
        nibabel.save(new_seg, MSSEG2_LABELS_DIR / label_file_name)
