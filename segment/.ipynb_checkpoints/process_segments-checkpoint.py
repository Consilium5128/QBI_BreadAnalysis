import numpy as np

from scipy.ndimage import distance_transform_edt, label, find_objects
from skimage.measure import regionprops, regionprops_table

def merge_regions(labelled_image, distance_transform, distance_threshold):
    """
    Merge regions in a labelled image based on a distance threshold.
    """
    regions = regionprops(labelled_image)
    merged_image = np.copy(labelled_image)
    label_pairs_to_merge = set()
    region_distance = np.zeros(labelled_image.shape)
    count=0
    
    # Find pairs of labels to merge based on distance threshold
    for region in regions:
        slice_obj = find_objects(labelled_image == region.label)[0]
        region_distance[slice_obj] = distance_transform[slice_obj]
        region_coords = region.coords
        
        for coord in region_coords:
            #print(region_distance, tuple(coord))
            if region_distance[tuple(coord)] < distance_threshold:
                neighbors = get_neighbors(tuple(coord), labelled_image.shape)
                for neighbor in neighbors:
                    if labelled_image[neighbor] > 0 and labelled_image[neighbor] != region.label:
                        label_pairs_to_merge.add((region.label, labelled_image[neighbor]))
                        count+=1
    
    print("Merging")
    # Merge the regions
    for label1, label2 in label_pairs_to_merge:
        merged_image[merged_image == label2] = label1
    
    # Relabel the merged image to maintain sequential labels
    merged_image, _ = label(merged_image > 0)
    
    return merged_image, count

def filter_regions_by_size(labelled_image, min_size):
    """
    Filter regions in a labelled image based on a minimum size threshold.
    """
    regions = regionprops(labelled_image)
    filtered_labelled_image = np.zeros_like(labelled_image)
    new_label = 1
    count = 0
    
    for region in regions:
        if (region.area >= min_size and (region.area<(labelled_image.size*0.80))):
            for coord in region.coords:
                filtered_labelled_image[coord[0], coord[1], coord[2]] = new_label
            new_label += 1
            count+=1
            
    print(count)
    return filtered_labelled_image, count


def get_neighbors(coord, shape):
    """
    Get neighbors of a voxel in a 3D image.
    """
    neighbors = []
    for i in range(coord[0] - 1, coord[0] + 2):
        for j in range(coord[1] - 1, coord[1] + 2):
            for k in range(coord[2] - 1, coord[2] + 2):
                #print(i, j, k, shape[0], shape[1], shape[2])
                if (i==0 and j==0 and k==0):
                    continue
                elif (0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]):
                    neighbors.append((i, j, k))
    return neighbors