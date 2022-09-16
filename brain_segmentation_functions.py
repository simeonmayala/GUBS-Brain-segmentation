

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import shortest_path, dijkstra
from scipy.sparse.csgraph import breadth_first_tree
from scipy.sparse import csr_matrix
from sknetwork.path import shortest_path
from skimage import color
import numpy as np
from skimage.filters import gaussian

def graph_ingredients(image, brain_nodes, skull_nodes, out_skull_nodes):
    segments = np.arange(np.prod(image.shape), dtype=np.int).reshape(image.shape)
    
    # compute edges weights
    zdirection = abs((image[:,:,1:] - image[:,:,:-1]).ravel())
    ydirection = abs((image[:,1:,:] - image[:,:-1,:]).ravel())
    xdirection = abs((image[1:,:,:] - image[:-1,:,:]).ravel())
    costs = np.concatenate((xdirection, ydirection, zdirection ))+1
    
    #Compute edges
    y_down  = segments[:-1,:,:].ravel()
    x_down  = segments[1:,:,:].ravel()
    y_right = segments[:,:-1,:].ravel()
    x_right = segments[:,1:,:].ravel()
    z_down  = segments[:,:,:-1].ravel()
    z_right = segments[:,:,1:].ravel()
    rows = np.concatenate((y_down, y_right, z_right))
    cols = np.concatenate((x_down, x_right, z_down))

    # collapsing nodes
    rows[np.isin(rows, brain_nodes)] = brain_nodes[0]
    cols[np.isin(cols, brain_nodes)] = brain_nodes[0]

    rows[np.isin(rows, skull_nodes)] = skull_nodes[0]
    cols[np.isin(cols, skull_nodes)] = skull_nodes[0]

    rows[np.isin(rows, out_skull_nodes)] = out_skull_nodes[0]
    cols[np.isin(cols, out_skull_nodes)] = out_skull_nodes[0]
  
    row_extension = np.full(len(brain_nodes), brain_nodes[0])
    rows = np.concatenate((rows, row_extension))
    cols = np.concatenate((cols, brain_nodes))
    costs = np.concatenate((costs, np.ones(len(brain_nodes))))

    row_extension_end = np.full(len(skull_nodes), skull_nodes[0])
    rows = np.concatenate((rows, row_extension_end))
    cols = np.concatenate((cols, skull_nodes))
    costs = np.concatenate((costs, np.ones(len(skull_nodes))))

    row_extension_out = np.full(len(out_skull_nodes), out_skull_nodes[0])
    rows = np.concatenate((rows, row_extension_out))
    cols = np.concatenate((cols, out_skull_nodes))
    costs = np.concatenate((costs, np.ones(len(out_skull_nodes))))

    mask = cols!=rows
    cols = cols[mask]
    rows = rows[mask]
    costs = costs[mask]
    
    return cols, rows, costs

def mst_construction(cols, rows, costs, image):
    segments = np.arange(np.prod(image.shape), dtype=np.int).reshape(image.shape)
    # construct a graph
    create_sparse_gr = csr_matrix((costs, (rows, cols)), shape = (np.prod(segments.shape), np.prod(segments.shape)))
    # construct a minimum spanning tree
    mst = minimum_spanning_tree(create_sparse_gr)
    mst = mst + mst.T
    return mst

def BFS_path(mst, node1, node2):
    BFS_tree = breadth_first_tree(mst, node1, directed=False)
    path = shortest_path(BFS_tree, sources = node1, targets=node2)
    return path

def split_components(mst, node1, node2):
    
    path_finder = np.array(BFS_path(mst, node1, node2))
    if len(path_finder)==0:
        return connected_components(mst)[1]
    sources = path_finder[:-1]
    targets = path_finder[1:]
    longest_ind_edge = np.argmax(mst[sources].T[targets].diagonal())
    n1 = sources[longest_ind_edge]
    n2 = targets[longest_ind_edge]

    mst[n1,n2]=0
    mst[n2,n1]=0
    mst.eliminate_zeros()
 
    return connected_components(mst)[1]

def unpadding(data1, thickness):
    thick = thickness
    thickh = int(thick/2)
    xend1, yend1, zend1 = data1.shape
    xend, yend, zend = xend1 -thick, yend1-thick, zend1-thick
    xlim, ylim, zlim = xend1-thickh, yend1-thickh, zend1-thickh
    data = data1[thickh:xlim, thickh:ylim,thickh:zlim]
    return data
