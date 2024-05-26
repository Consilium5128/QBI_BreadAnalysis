import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from scipy.ndimage import label
from skimage.measure import regionprops, regionprops_table

def perform_kmeans_clustering_3d(image_3d, n_clusters=2):
    """
    Perform K-means clustering on a 3D image.

    Parameters:
    image_3d (np.ndarray): 3D image array.
    n_clusters (int): Number of clusters.

    Returns:
    np.ndarray: 3D array with the same shape as image_3d, containing cluster labels.
    """
    # Reshape the 3D image into a 2D array (samples x features)
    depth, height, width = image_3d.shape
    image_2d = image_3d.reshape(-1, 1)  # Assuming grayscale image (1 feature)
    print(image_2d.shape)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(image_2d)
    labels_1d = kmeans.labels_

    # Reshape the 1D labels array back into the 3D image shape
    labels_3d = labels_1d.reshape(depth, height, width)
    print(labels_3d.shape)
    print(labels_3d.max())

    return labels_3d

def extract_clusters_3d(img, downsample_img, img_sample):
    lbl, num = label(img)
    lbl_regions = pd.DataFrame(regionprops_table(lbl, properties=('label', 'area')))
    max_area_index = lbl_regions['area'].idxmax()
    largest_area_label = lbl_regions.loc[max_area_index, 'label']
    #largest_area_label = lbl_regions.iloc[1]['label']
    print("Largest label = "+str(largest_area_label))
    check = (sum(sum(sum(lbl==largest_area_label)))+sum(sum(sum(lbl==0))))/lbl.size
    
    fig,ax=plt.subplots(1,4,figsize=(20,5))
    ax[0].imshow(downsample_img[img_sample],interpolation='none'), ax[0].axis('off')
    ax[0].set_title('Bilevel image');
    ax[1].imshow(lbl[img_sample], interpolation='none'),ax[1].axis('off')
    ax[1].set_title('Labeled items');
    _ = ax[2].hist(lbl.ravel(), range=(0,50), bins=51)
    ax[2].set_title('Label Histogram')
    ax[2].set_xlabel('Image value'), ax[0].set_ylabel('Number of pixels')
    
    img_1 = lbl==largest_area_label
    if check<0.90:
        print(str(check)+" Incorrect number of clusters :"+str(lbl.max()))
        return img_1
    elif ((check<0.98) and (check>=0.9)):
        print("Check labels")
        return img_1
    else:
        print(check)
        ax[3].imshow(img_1[img_sample],interpolation='none'), ax[3].axis('off')
        ax[3].set_title('1');
        return img_1