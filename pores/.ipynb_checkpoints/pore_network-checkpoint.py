import numpy as np
import networkx as nx

from scipy.ndimage import label
from skimage.morphology import skeletonize_3d
from skimage.measure import regionprops, regionprops_table

def analyze_pore_network_connectivity(binary_image_stack):
    """
    Analyze the pore network connectivity of a 3D binary image stack.
    """
    skeleton = skeletonize_3d(binary_image_stack)
    labeled_skeleton, num_features = label(skeleton)
    
    graph = nx.Graph()
    
    regions = regionprops(labeled_skeleton)
    
    for region in regions:
        coords = region.coords
        for coord in coords:
            graph.add_node(tuple(coord))
            for neighbor in get_neighbors(coord, labeled_skeleton.shape):
                if labeled_skeleton[neighbor] == region.label:
                    graph.add_edge(tuple(coord), neighbor)
    
    num_connected_components = nx.number_connected_components(graph)
    return num_connected_components, graph