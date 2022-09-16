
import numpy as np
from skimage import color
from skimage.filters import threshold_otsu
from skimage import morphology
import scipy.ndimage as ndimage
from skimage.morphology import  erosion, ball

def remove_small_objects(img, rd):    
    binary1 = img.copy()
    tresh_b1 = threshold_otsu(binary1)
    binary = binary1 > tresh_b1  
    labels = morphology.label(binary)
    labels_num = [len(labels[labels==each]) for each in np.unique(labels)]
    rank = np.argsort(np.argsort(labels_num))
    index = list(rank).index(len(rank)-2)
    new_img = img.copy()
    new_img[labels!=index] = 0
    temp = new_img.copy()
    temp_bin = temp>0
    B = ndimage.binary_closing(temp_bin, structure=np.ones((rd,rd,rd))).astype(int)
    binary1[B==0] = 0
    return binary1, B

def padding(data1, thickness):
    data = data1.copy()
    thick = thickness
    thickh = int(thick/2)
    xend1, yend1, zend1 = data1.shape
    xend, yend, zend = xend1 +thick, yend1+thick, zend1+thick
    data = np.zeros((xend, yend, zend))
    xlim, ylim, zlim = xend1+thickh, yend1+thickh, zend1+thickh
    data[thickh:xlim, thickh:ylim,thickh:zlim] = data1
    return data

def unpadding(data1, thickness):
    thick = thickness
    thickh = int(thick/2)
    xend1, yend1, zend1 = data1.shape
    xend, yend, zend = xend1 -thick, yend1-thick, zend1-thick
    xlim, ylim, zlim = xend1-thickh, yend1-thickh, zend1-thickh
    data = data1[thickh:xlim, thickh:ylim,thickh:zlim]
    return data


def opening_parts(img_mod, BINARY_top, BINARY_lower, ptsxe, ptsye, ptsze):
    binarized_bin = BINARY_top.copy()
    rad = 2
    for p in range(len(ptsxe)):
        r, c, h = ptsxe[p], ptsye[p], ptsze[p]
        roi = binarized_bin[r-rad:r+rad, c-rad:c+rad, h-rad:h+rad]
        roi[:] = erosion(roi, ball(1))
        
    image_top = img_mod.copy()
    image_top[binarized_bin==0]=0
    
    image_lower = img_mod.copy()
    image_lower[BINARY_lower==0]=0
    
    neck_threshold = 0.28
    img_lower_hem = image_lower.copy()
    img_lower_hem[img_lower_hem < neck_threshold] = 0
    
    alpha = 6
    recons_image = image_top + img_lower_hem
    recons_mod = recons_image.copy()
    roi = recons_mod[:, 90:, :]
    roi[:] = roi[roi<0.3]=0
    
    return recons_image
