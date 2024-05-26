import numpy as np

from skimage.feature import peak_local_max

def thickness_map_3d(img_dist, min_distance=5):
    pore_candidates = peak_local_max(img_dist, min_distance=min_distance)
    print('Found', len(pore_candidates), 'bubbles')

    thickness_map = np.zeros(img_dist.shape, dtype=np.float32)
    xx, yy, zz = np.meshgrid(np.arange(img_dist.shape[1]),
                             np.arange(img_dist.shape[0]),
                             np.arange(img_dist.shape[2])
                             )
    # sort candidates by size
    sorted_candidates = sorted(
        pore_candidates, key=lambda xyz: img_dist[tuple(xyz)])
    for label_idx, (x, y, z) in enumerate(sorted_candidates):
        cur_pore_radius = img_dist[x, y, z]
        cur_pore = ((xx-float(y))**2 +
                      (yy-float(x))**2 +
                      (zz-float(z))**2) <= cur_pore_radius**2
        thickness_map[cur_pore] = cur_pore_radius
        
    return thickness_map

def compute_distance_from_centroid(regions, coord, slicenum=256):
    try:
        return np.sqrt((regions["centroid-0"][0] - coord[0])**2 + (regions["centroid-1"][0] - coord[1])**2 + (regions["centroid-2"][0] - coord[2])**2)
    except:
        return np.sqrt(((slicenum/2) - coord[0])**2 + (regions["centroid-0"][0] - coord[1])**2 + (regions["centroid-1"][0] - coord[2])**2)

def compute_distance_from_centroid_zconstrain(regions, coord):
    # Assumption - x and y coords of centroid are true for each slice
    try:
        return np.sqrt((regions["centroid-1"][0] - coord[1])**2 + (regions["centroid-2"][0] - coord[2])**2)
    except:
        return np.sqrt((regions["centroid-0"][0] - coord[1])**2 + (regions["centroid-1"][0] - coord[2])**2)