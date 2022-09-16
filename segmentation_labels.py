
import numpy as np
from skimage import color
from skimage.filters import threshold_otsu
from skimage import morphology
import scipy.ndimage as ndimage
from scipy.spatial import distance
from scipy.ndimage import distance_transform_edt
import random

def remove_sobj(img, d): 
    binary = img.copy()
    roi = binary[:, :, :]
    roi[:] = ndimage.binary_opening(roi, structure=np.ones((d,d,d))).astype(int)
    labels = morphology.label(binary)
    labels_num = [len(labels[labels==each]) for each in np.unique(labels)]
    rank = np.argsort(np.argsort(labels_num))
    index = list(rank).index(len(rank)-2)
    new_img = img.copy()
    new_img[labels!=index] = 0
    return new_img

def expand_labels(label_image, distance=1):

    distances, nearest_label_coords = distance_transform_edt(
        label_image == 0, return_indices=True)
    
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords]
    
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out

