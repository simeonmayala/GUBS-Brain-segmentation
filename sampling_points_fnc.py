
import numpy as np
from skimage import color
from skimage.filters import threshold_otsu
from skimage import morphology
import scipy.ndimage as ndimage
from scipy.spatial import distance
import random

def sampling_points_within_brain(ptsx, ptsy, ptsz, template):
    midpx, midpy, midpz = int(np.mean(ptsx)), int(np.mean(ptsy)), int(np.mean(ptsz))
    indices = []
    a = (midpx, midpy, midpz)
    for i in range(len(ptsx)):
        b = (ptsx[i], ptsy[i], ptsz[i])
        dist = distance.euclidean(a, b)
        if dist > 35:
            indices.append(i)
    for i in sorted(indices, reverse=True):
        del ptsx[i]
        del ptsy[i]
        del ptsz[i]
    nodes_within_brain = [template[ptsx[i], ptsy[i], ptsz[i]] for i in range(len(ptsx))]
    return nodes_within_brain

def sampling_points_within_nonbrain(ptsx_out, ptsy_out, ptsz_out, template, recons_image):
    temp_indices = []
    for i in range(len(ptsz_out)):
        value = recons_image[ptsx_out[i],ptsy_out[i], ptsz_out[i]]
        if value == 0:
            temp_indices.append(i)
    for i in sorted(temp_indices, reverse=True):
        del ptsx_out[i]
        del ptsy_out[i]
        del ptsz_out[i]
    nodes_within_skull = [template[ptsx_out[i], ptsy_out[i], ptsz_out[i]] for i in range(len(ptsx_out))]
    return nodes_within_skull

def sampling_points_within_background(BINARY, alpha, template):
    pts_out_head = unpadding(BINARY, alpha)
    lx, ly, lz = pts_out_head.shape
    lx, ly, lz = lx-1, ly-1, lz-1
    points_out_head1 = np.where(pts_out_head[0,:,:]==0)
    points_out_head2 = np.where(pts_out_head[:,0,:]==0)
    points_out_head3 = np.where(pts_out_head[:,:,0]==0)
    points_out_head4 = np.where(pts_out_head[lx,:,:]==0)
    points_out_head5 = np.where(pts_out_head[:,ly,:]==0)
    points_out_head6 = np.where(pts_out_head[:,:,lz]==0)
    phase1y, phase1z = list(points_out_head1[0]), list(points_out_head1[1])
    phase2x, phase2z = list(points_out_head2[0]), list(points_out_head2[1])
    phase3x, phase3y = list(points_out_head3[0]), list(points_out_head3[1])
    phase4y, phase4z = list(points_out_head4[0]), list(points_out_head4[1])
    phase5x, phase5z = list(points_out_head5[0]), list(points_out_head5[1])
    phase6x, phase6y = list(points_out_head6[0]), list(points_out_head6[1])
    points_phase1 = [template[0,phase1y[i], phase1z[i]] for i in range(len(phase1y))]
    points_phase2 = [template[phase2x[i], 0, phase2z[i]] for i in range(len(phase2x))]
    points_phase3 = [template[phase3x[i], phase3y[i], 0] for i in range(len(phase3x))]
    points_phase4 = [template[lx,phase4y[i], phase4z[i]] for i in range(len(phase4y))]
    points_phase5 = [template[phase5x[i], ly, phase5z[i]] for i in range(len(phase5x))]
    points_phase6 = [template[phase6x[i], phase6y[i], lz] for i in range(len(phase6x))]
    
    nodes_out_head = points_phase1+points_phase2+points_phase3+points_phase4+points_phase5+points_phase6
    return nodes_out_head 

def randomize_nodes(nodes_within_brain, nodes_within_skull, nodes_out_head):
    
    brain_nodes = [] # within the bran 
    for i in range(len(nodes_within_brain)):
        ind = random.randint(0, len(nodes_within_brain)-1)
        node = nodes_within_brain[ind]
        brain_nodes.append(node)
        
    skull_nodes = [] # within the non-brain tissues
    for i in range(len(nodes_within_skull)):
        ind = random.randint(0, len(nodes_within_skull)-1)
        node = nodes_within_skull[ind]
        skull_nodes.append(node)
        
    out_skull_nodes = [] # In the background
    for i in range(20000):
        ind = random.randint(0, len(nodes_out_head)-1)
        node = nodes_out_head[ind]
        out_skull_nodes.append(node)
        
    return brain_nodes, skull_nodes, out_skull_nodes

def unpadding(data1, thickness):
    thick = thickness
    thickh = int(thick/2)
    xend1, yend1, zend1 = data1.shape
    xend, yend, zend = xend1 -thick, yend1-thick, zend1-thick
    xlim, ylim, zlim = xend1-thickh, yend1-thickh, zend1-thickh
    data = data1[thickh:xlim, thickh:ylim,thickh:zlim]
    return data
