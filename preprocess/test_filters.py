import numpy as np
from scipy.ndimage.filters import median_filter, uniform_filter, gaussian_filter, maximum_filter, minimum_filter
import matplotlib.pyplot as plt

def give_snr_2d(img, bounding_box):
    subA=img[bounding_box[1,0]:bounding_box[1,1],bounding_box[0,0]:bounding_box[0,1]];
    snrA=np.mean(subA)/np.std(subA)
    return snrA, subA

def show_test_filter_sizes_2d(img, sizes, bounding_box, filter_type='median'):
    fig, ax = plt.subplots(1, 1+len(sizes), figsize=((len(sizes)*3),20))
    snr_img, _ = give_snr_2d(img, bounding_box)

    ax[0].imshow(img,cmap='gray')
    ax[0].set_title("Raw, SNR:{0:2f}".format(snr_img))
    for idx,filter_size in enumerate(sizes):       

        if filter_type=='median':
            filtered = median_filter(img,filter_size) # Here the actual filtering takes place
        elif filter_type=='uniform':
            filtered = uniform_filter(img,filter_size)
        elif filter_type=='gaussian':
            filtered = gaussian_filter(img,filter_size)
        elif filter_type=='maximum':
            filtered = maximum_filter(img,filter_size)
        elif filter_type=='minimum':
            filtered = minimum_filter(img,filter_size)
        
        snr, _ = give_snr_2d(filtered, bounding_box)

        ax[1+idx].imshow(filtered, cmap='gray', interpolation='none')
        ax[1+idx].set_title("size:{}, SNR:{:.2f}".format(filter_size,snr))