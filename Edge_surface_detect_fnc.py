
import numpy as np
from skimage import color
from skimage.filters import threshold_otsu
from skimage import morphology
import scipy.ndimage as ndimage

def separate_head_scan(img_mod, ratio):
    upper_hemisphere = img_mod.copy()
    upper_hemisphere[:,:,:ratio] = 0
    lower_hemisphere = img_mod.copy()
    lower_hemisphere[:,:,ratio:] = 0
    return upper_hemisphere, lower_hemisphere


def identify_points(Binarized, img_org, delta, alpha):
        # reduce the size of the image back to its original size
#     alpha = 6
    bin_top_hem = unpadding(Binarized, alpha)
    img_top     = unpadding(img_org, alpha)
    template = np.arange(np.prod(img_top.shape), dtype=np.int).reshape(img_top.shape)
    
    der1top = np.diff(bin_top_hem, axis = -1)
    der2top = np.diff(bin_top_hem, axis =  0)
    der3top = np.diff(bin_top_hem, axis =  1)
    
    # points detection for axis ==0
    pos_dev0top = np.where(der2top==1)
    neg_dev0top = np.where(der2top==-1)

    # points detection for axis ==1
    pos_dev1top = np.where(der3top==1)
    neg_dev1top = np.where(der3top==-1)

    # points detection for axis ==-1
    pos_dev_neg1top = np.where(der1top==1)
    neg_dev_neg1top = np.where(der1top==-1)
    # parameter tuning
    X = list(pos_dev0top[0]) + list(neg_dev0top[0]) 
    Y = list(pos_dev0top[1]) + list(neg_dev0top[1]) 
    Z = list(pos_dev0top[2]) + list(neg_dev0top[2]) 
    # concatenate points from axis == 1
    X1 = list(pos_dev1top[0]) + list(neg_dev1top[0]) 
    Y1 = list(pos_dev1top[1]) + list(neg_dev1top[1]) 
    Z1 = list(pos_dev1top[2]) + list(neg_dev1top[2])
    xd, yd, zd = list(neg_dev_neg1top[0]), list(neg_dev_neg1top[1]),list(neg_dev_neg1top[2])

    if delta ==5:
        selected_points_x = X + X1 + xd
        selected_points_y = Y + Y1 + yd
        selected_points_z = Z + Z1 + zd
        
    else:
        selected_points_x = X + X1 
        selected_points_y = Y + Y1 
        selected_points_z = Z + Z1 
    
    h_limit = int((np.min(selected_points_z)+np.max(selected_points_z))/2)
    y_limit = int((np.min(selected_points_y)+np.max(selected_points_y))/2)
    x_limit = int((np.min(selected_points_x)+np.max(selected_points_x))/2)
    
   
    ptsx, ptsy, ptsz = [],[],[]
    for i in range(1): 
        arx = [val+delta if val <x_limit else val-delta for val in selected_points_x]
        ary = [val+delta if val <y_limit else val-delta for val in selected_points_y]
        arz = [val-delta for val in selected_points_z]
        ptsx = ptsx + arx
        ptsy = ptsy + ary
        ptsz = ptsz + arz
        delta +=1    
    return ptsx, ptsy, ptsz, template, bin_top_hem

def unpadding(data1, thickness):
    thick = thickness
    thickh = int(thick/2)
    xend1, yend1, zend1 = data1.shape
    xend, yend, zend = xend1 -thick, yend1-thick, zend1-thick
    xlim, ylim, zlim = xend1-thickh, yend1-thickh, zend1-thickh
    data = data1[thickh:xlim, thickh:ylim,thickh:zlim]
    return data
