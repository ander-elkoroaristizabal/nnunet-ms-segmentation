import os

import nibabel

from custom_scripts.A_config import MSSEG2_LABELS_DIR

label_files = os.listdir(MSSEG2_LABELS_DIR)
for label_file_name in label_files:
    label_file = nibabel.load(MSSEG2_LABELS_DIR / label_file_name)
    seg = label_file.get_fdata()
    seg[seg == 1] = 2
    new_seg = nibabel.Nifti1Image(seg, label_file.affine, label_file.header)
    nibabel.save(new_seg, MSSEG2_LABELS_DIR / label_file_name)
