from typing import Tuple, Union

import numpy as np
from matplotlib import colors, pyplot as plt


def axes_plot(sample_im: np.ndarray, slices: Union[Tuple[int], None] = None):
    """Plot all three axes of a sample image. If 'slices', use custom slices."""
    if not slices:
        im_shape = sample_im.shape
        slices = (im_shape[0] // 2, im_shape[1] // 2, im_shape[2] // 2)
    # Axes plot:
    f, (p1, p2, p3) = plt.subplots(1, 3, figsize=(9, 6), gridspec_kw={'width_ratios': [1, 210 / 180, (210 / 180) ** 2]})
    # Sagittal plane:
    _ = p1.imshow(sample_im[slices[0], :, :], cmap='gray')
    p1.set_title('Sagittal plane')
    # Coronal plane:
    _ = p2.imshow(sample_im[:, slices[1], :], cmap='gray')
    p2.set_title('Coronal plane')
    # Axial plane:
    _ = p3.imshow(sample_im[:, :, slices[2]], cmap='gray')
    p3.set_title('Axial plane')


def get_limits(central_voxel: Tuple[int], patch_size: int, im_shape: Tuple[int]):
    """Get the limits of the patch of size 'patch_size' around 'central_voxel' and within image limits."""
    # Initial limits:
    s_limits = [central_voxel[0] - patch_size, central_voxel[0] + patch_size]
    c_limits = [central_voxel[1] - patch_size, central_voxel[1] + patch_size]
    a_limits = [central_voxel[2] - patch_size, central_voxel[2] + patch_size]
    # Fitting them to the image size:
    for i, limits in enumerate([s_limits, c_limits, a_limits]):
        limits[0] = max(limits[0], 0)
        limits[1] = min(limits[1], im_shape[i])
    return s_limits, c_limits, a_limits


def plot_basal_lesion(b_im, fu_im, labels, central_voxel=None, patch_size: int = 25):
    """Plot basal lesions in a patch of size 'patch_size' around the 'central_voxel'."""
    # Masking others:
    only_basal_lesions = np.ma.masked_where(labels != 1, labels)
    orange = colors.ListedColormap(['darkorange'])
    # Limits:
    s_limits, c_limits, a_limits = get_limits(
        central_voxel=central_voxel,
        patch_size=patch_size,
        im_shape=b_im.shape
    )
    # Plot:
    f, (r1, r2, r3) = plt.subplots(3, 3, figsize=(10, 12))
    # Plano sagital:
    s_zoom = np.s_[central_voxel[0], c_limits[0]:c_limits[1], a_limits[0]: a_limits[1]]
    r1[0].set_ylabel("SAGITTAL PLANE")
    # Basal
    _ = r1[0].imshow(b_im[s_zoom], cmap='gray')
    r1[0].set_title('Basal image')
    # Basal w. Mask
    _ = r1[2].imshow(b_im[s_zoom], cmap='gray')
    _ = r1[2].imshow(only_basal_lesions[s_zoom], cmap=orange)
    r1[2].set_title('Basal image w. basal lesions')
    # Follow up
    _ = r1[1].imshow(fu_im[s_zoom], cmap='gray')
    r1[1].set_title('Follow-up image')
    # Plano coronal:
    c_zoom = np.s_[s_limits[0]:s_limits[1], central_voxel[1], a_limits[0]: a_limits[1]]
    r2[0].set_ylabel("CORONAL PLANE")
    # Basal
    _ = r2[0].imshow(b_im[c_zoom], cmap='gray')
    r2[0].set_title('Basal image')
    # Basal w mask:
    _ = r2[2].imshow(b_im[c_zoom], cmap='gray')
    _ = r2[2].imshow(only_basal_lesions[c_zoom], cmap=orange)
    r2[2].set_title('Basal image w. basal lesions')
    # Follow up
    _ = r2[1].imshow(fu_im[c_zoom], cmap='gray')
    r2[1].set_title('Follow-up image')
    # Plano axial:
    a_zoom = np.s_[s_limits[0]:s_limits[1], c_limits[0]: c_limits[1], central_voxel[2]]
    r3[0].set_ylabel("AXIAL PLANE")
    # Basal
    _ = r3[0].imshow(b_im[a_zoom], cmap='gray')
    r3[0].set_title('Basal image')
    # Basal with mask
    _ = r3[2].imshow(b_im[a_zoom], cmap='gray')
    _ = r3[2].imshow(only_basal_lesions[a_zoom], cmap=orange)
    r3[2].set_title('Basal image w. basal lesions')
    # Follow up
    _ = r3[1].imshow(fu_im[a_zoom], cmap='gray')
    r3[1].set_title('Follow-up image')
    return f


def plot_new_lesion(b_im, fu_im, labels, central_voxel, patch_size: int = 25):
    """Plot new lesions in a patch of size 'patch_size' around the 'central_voxel'."""
    only_new_lesions = np.ma.masked_where(labels != 2, labels)
    red = colors.ListedColormap(['red'])
    # Limits:
    s_limits, c_limits, a_limits = get_limits(
        central_voxel=central_voxel,
        patch_size=patch_size,
        im_shape=b_im.shape
    )
    # Plot:
    f, (r1, r2, r3) = plt.subplots(3, 3, figsize=(10, 12))
    # Plano sagital:
    s_zoom = np.s_[central_voxel[0], c_limits[0]:c_limits[1], a_limits[0]: a_limits[1]]
    r1[0].set_ylabel("SAGITTAL PLANE")
    # Basal
    _ = r1[0].imshow(b_im[s_zoom], cmap='gray')
    r1[0].set_title('Basal image')
    # Follow up
    _ = r1[1].imshow(fu_im[s_zoom], cmap='gray')
    r1[1].set_title('Follow-up image')
    # Follow-up w. Mask
    _ = r1[2].imshow(fu_im[s_zoom], cmap='gray')
    _ = r1[2].imshow(only_new_lesions[s_zoom], cmap=red)
    r1[2].set_title('Follow-up image w. new lesions')
    # Plano coronal:
    c_zoom = np.s_[s_limits[0]:s_limits[1], central_voxel[1], a_limits[0]: a_limits[1]]
    r2[0].set_ylabel("CORONAL PLANE")
    # Basal
    _ = r2[0].imshow(b_im[c_zoom], cmap='gray')
    r2[0].set_title('Basal image')
    # Follow up
    _ = r2[1].imshow(fu_im[c_zoom], cmap='gray')
    r2[1].set_title('Follow-up image')
    # Follow-up w mask:
    _ = r2[2].imshow(fu_im[c_zoom], cmap='gray')
    _ = r2[2].imshow(only_new_lesions[c_zoom], cmap=red)
    r2[2].set_title('Follow-up image w. new lesions')
    # Plano axial:
    a_zoom = np.s_[s_limits[0]:s_limits[1], c_limits[0]: c_limits[1], central_voxel[2]]
    r3[0].set_ylabel("AXIAL PLANE")
    # Basal
    _ = r3[0].imshow(b_im[a_zoom], cmap='gray')
    r3[0].set_title('Basal image')
    # Follow up
    _ = r3[1].imshow(fu_im[a_zoom], cmap='gray')
    r3[1].set_title('Follow-up image')
    # Follow-up with mask
    _ = r3[2].imshow(fu_im[a_zoom], cmap='gray')
    _ = r3[2].imshow(only_new_lesions[a_zoom], cmap=red)
    r3[2].set_title('Follow-up image w. new lesions')
    return f


def plot_both_lesions(b_im, fu_im, labels, central_voxel, patch_size: int = 25):
    """Plot basal and new lesions in a patch of size 'patch_size' around the 'central_voxel'."""
    only_new_lesions = np.ma.masked_where(labels == 0, labels)
    both = colors.ListedColormap(['darkorange', 'red'])
    # Limits:
    s_limits, c_limits, a_limits = get_limits(
        central_voxel=central_voxel,
        patch_size=patch_size,
        im_shape=b_im.shape
    )
    # Plot:
    f, (r1, r2, r3) = plt.subplots(3, 3, figsize=(10, 12))
    # Plano sagital:
    s_zoom = np.s_[central_voxel[0], c_limits[0]:c_limits[1], a_limits[0]: a_limits[1]]
    r1[0].set_ylabel("SAGITTAL PLANE")
    # Basal
    _ = r1[0].imshow(b_im[s_zoom], cmap='gray')
    r1[0].set_title('Basal image')
    # Follow up
    _ = r1[1].imshow(fu_im[s_zoom], cmap='gray')
    r1[1].set_title('Follow-up image')
    # Follow-up w. Mask
    _ = r1[2].imshow(fu_im[s_zoom], cmap='gray')
    _ = r1[2].imshow(only_new_lesions[s_zoom], cmap=both, vmin=1, vmax=2)
    r1[2].set_title('Follow-up image w. \nbasal and new lesions')
    # Plano coronal:
    c_zoom = np.s_[s_limits[0]:s_limits[1], central_voxel[1], a_limits[0]: a_limits[1]]
    r2[0].set_ylabel("CORONAL PLANE")
    # Basal
    _ = r2[0].imshow(b_im[c_zoom], cmap='gray')
    r2[0].set_title('Basal image')
    # Follow up
    _ = r2[1].imshow(fu_im[c_zoom], cmap='gray')
    r2[1].set_title('Follow-up image')
    # Follow-up w mask:
    _ = r2[2].imshow(fu_im[c_zoom], cmap='gray')
    _ = r2[2].imshow(only_new_lesions[c_zoom], cmap=both, vmin=1, vmax=2)
    r2[2].set_title('Follow-up image w. \nbasal and new lesions')
    # Plano axial:
    a_zoom = np.s_[s_limits[0]:s_limits[1], c_limits[0]: c_limits[1], central_voxel[2]]
    r3[0].set_ylabel("AXIAL PLANE")
    # Basal
    _ = r3[0].imshow(b_im[a_zoom], cmap='gray')
    r3[0].set_title('Basal image')
    # Follow up
    _ = r3[1].imshow(fu_im[a_zoom], cmap='gray')
    r3[1].set_title('Follow-up image')
    # Follow-up with mask
    _ = r3[2].imshow(fu_im[a_zoom], cmap='gray')
    _ = r3[2].imshow(only_new_lesions[a_zoom], cmap=both, vmin=1, vmax=2)
    r3[2].set_title('Follow-up image w. \nbasal and new lesions')
    return f


def plot_basal_lesion_eval(b_im, fu_im, conf_mat, central_voxel, patch_size: int = 25):
    """Plot basal lesion detection results in a patch of size 'patch_size' around the 'central_voxel'."""
    # TP is green, FP is red, FN is blue
    cmap = colors.ListedColormap(['green', 'tab:red', 'blue'])
    # Limits:
    s_limits, c_limits, a_limits = get_limits(
        central_voxel=central_voxel,
        patch_size=patch_size,
        im_shape=b_im.shape
    )
    # Plot:
    f, (r1, r2, r3) = plt.subplots(3, 3, figsize=(10, 12))
    # Plano sagital:
    s_zoom = np.s_[central_voxel[0], c_limits[0]:c_limits[1], a_limits[0]: a_limits[1]]
    r1[0].set_ylabel("SAGITTAL PLANE")
    # Basal
    _ = r1[0].imshow(b_im[s_zoom], cmap='gray')
    r1[0].set_title('Basal image')
    # Basal w. Mask
    _ = r1[2].imshow(b_im[s_zoom], cmap='gray')
    _ = r1[2].imshow(conf_mat[s_zoom], cmap=cmap, vmin=1, vmax=3)
    r1[2].set_title('Basal image w. new lesions')
    # Follow up
    _ = r1[1].imshow(fu_im[s_zoom], cmap='gray')
    r1[1].set_title('Follow-up image')
    # Plano coronal:
    c_zoom = np.s_[s_limits[0]:s_limits[1], central_voxel[1], a_limits[0]: a_limits[1]]
    r2[0].set_ylabel("CORONAL PLANE")
    # Basal
    _ = r2[0].imshow(b_im[c_zoom], cmap='gray')
    r2[0].set_title('Basal image')
    # Basal w mask:
    _ = r2[2].imshow(b_im[c_zoom], cmap='gray')
    _ = r2[2].imshow(conf_mat[c_zoom], cmap=cmap, vmin=1, vmax=3)
    r2[2].set_title('Basal image w. new lesions')
    # Follow up
    _ = r2[1].imshow(fu_im[c_zoom], cmap='gray')
    r2[1].set_title('Follow-up image')
    # Plano axial:
    a_zoom = np.s_[s_limits[0]:s_limits[1], c_limits[0]: c_limits[1], central_voxel[2]]
    r3[0].set_ylabel("AXIAL PLANE")
    # Basal
    _ = r3[0].imshow(b_im[a_zoom], cmap='gray')
    r3[0].set_title('Basal image')
    # Basal with mask
    _ = r3[2].imshow(b_im[a_zoom], cmap='gray')
    _ = r3[2].imshow(conf_mat[a_zoom], cmap=cmap, vmin=1, vmax=3)
    r3[2].set_title('Basal image w. new lesions')
    # Follow up
    _ = r3[1].imshow(fu_im[a_zoom], cmap='gray')
    r3[1].set_title('Follow-up image')
    return f


def plot_new_lesion_eval(b_im, fu_im, conf_mat, central_voxel, patch_size: int = 25):
    """Plot new lesion detection results in a patch of size 'patch_size' around the 'central_voxel'."""
    # TP is green, FP is red, FN is blue
    cmap = colors.ListedColormap(['green', 'red', 'blue'])
    # Limits:
    s_limits, c_limits, a_limits = get_limits(
        central_voxel=central_voxel,
        patch_size=patch_size,
        im_shape=b_im.shape
    )
    # Plot:
    f, (r1, r2, r3) = plt.subplots(3, 3, figsize=(10, 12))
    # Plano sagital:
    s_zoom = np.s_[central_voxel[0], c_limits[0]:c_limits[1], a_limits[0]: a_limits[1]]
    r1[0].set_ylabel("SAGITTAL PLANE")
    # Basal
    _ = r1[0].imshow(b_im[s_zoom], cmap='gray')
    r1[0].set_title('Basal image')
    # Follow up
    _ = r1[1].imshow(fu_im[s_zoom], cmap='gray')
    r1[1].set_title('Follow-up image')
    # Follow-up w. Mask
    _ = r1[2].imshow(fu_im[s_zoom], cmap='gray')
    _ = r1[2].imshow(conf_mat[s_zoom], cmap=cmap, vmin=1, vmax=3)
    r1[2].set_title('Follow-up image w. new lesions')
    # Plano coronal:
    c_zoom = np.s_[s_limits[0]:s_limits[1], central_voxel[1], a_limits[0]: a_limits[1]]
    r2[0].set_ylabel("CORONAL PLANE")
    # Basal
    _ = r2[0].imshow(b_im[c_zoom], cmap='gray')
    r2[0].set_title('Basal image')
    # Follow up
    _ = r2[1].imshow(fu_im[c_zoom], cmap='gray')
    r2[1].set_title('Follow-up image')
    # Follow-up w mask:
    _ = r2[2].imshow(fu_im[c_zoom], cmap='gray')
    _ = r2[2].imshow(conf_mat[c_zoom], cmap=cmap, vmin=1, vmax=3)
    r2[2].set_title('Follow-up image w. new lesions')
    # Plano axial:
    a_zoom = np.s_[s_limits[0]:s_limits[1], c_limits[0]: c_limits[1], central_voxel[2]]
    r3[0].set_ylabel("AXIAL PLANE")
    # Basal
    _ = r3[0].imshow(b_im[a_zoom], cmap='gray')
    r3[0].set_title('Basal image')
    # Follow up
    _ = r3[1].imshow(fu_im[a_zoom], cmap='gray')
    r3[1].set_title('Follow-up image')
    # Follow-up with mask
    _ = r3[2].imshow(fu_im[a_zoom], cmap='gray')
    _ = r3[2].imshow(conf_mat[a_zoom], cmap=cmap, vmin=1, vmax=3)
    r3[2].set_title('Follow-up image w. new lesions')
    return f
