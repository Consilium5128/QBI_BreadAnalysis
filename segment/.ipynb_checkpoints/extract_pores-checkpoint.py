import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries, watershed
from scipy.ndimage import distance_transform_edt, binary_fill_holes, label
from skimage.morphology import ball, binary_opening, binary_dilation
from skimage.filters import sobel

def extract_pores_3d(img, structure=np.ones((1,3,3))):
    filled_pores = binary_fill_holes(img, structure=structure)
    test_img_edge = sobel(filled_pores)
    inverted_image = np.copy(img)
    inverted_image[filled_pores] = inverted_image.max() ^ inverted_image[filled_pores]
    inv_img_test = inverted_image | (test_img_edge!=0)
    
    return inverted_image, inv_img_test, filled_pores

def segment_pores_watershed_3d(inverted_image_1, cur_img_sample):
    opened_inverted_image_1 = binary_opening(inverted_image_1)
    dilated_inverted_image_1 = binary_dilation(opened_inverted_image_1).astype(int)
    dilated_inverted_image_1*=255
    
    dist_transform_opened_1 = distance_transform_edt(opened_inverted_image_1)
    sure_fg_1 = (dist_transform_opened_1>(0.1*dist_transform_opened_1.max())).astype(int)
    sure_fg_1*=255
    unknown_1 = dilated_inverted_image_1 - sure_fg_1
    #print(sum(sum(sum(sure_fg_1)))/255)

    label_fg_1, _ = label(sure_fg_1)
    label_fg_1+=1
    label_fg_1[unknown_1==255] = 0

    watershed_fg_1 = watershed(inverted_image_1, markers=label_fg_1)
    inverted_image_1_test = np.copy(inverted_image_1)
    inverted_image_1_test[watershed_fg_1 == -1] = 255
    
    fig, ax = plt.subplots(1,6,figsize=(30,10))

    ax[0].imshow(dilated_inverted_image_1[cur_img_sample], cmap='gray'); ax[0].set_title("sure background")
    ax[1].imshow(sure_fg_1[cur_img_sample], cmap='gray'); ax[1].set_title("sure foreground")
    ax[2].imshow(unknown_1[cur_img_sample], cmap='gray'); ax[2].set_title("unknown")
    ax[3].imshow(label_fg_1[cur_img_sample], cmap='viridis'); ax[3].set_title("label")
    ax[4].imshow(watershed_fg_1[cur_img_sample], cmap='viridis'); ax[4].set_title("watershed")
    ax[5].imshow(inverted_image_1_test[cur_img_sample], cmap='viridis'); ax[5].set_title("inverted_test")
    
    return watershed_fg_1

def segment_pores_lbl_3d(inverted_image_1):
    inverted_image_dist_1 = distance_transform_edt(inverted_image_1)
    lbl_1, num_1 = label(inverted_image_1)
    
    return lbl_1, inverted_image_dist_1
    