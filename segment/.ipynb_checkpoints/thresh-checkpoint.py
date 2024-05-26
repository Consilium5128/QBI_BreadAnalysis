import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import dilation, opening, disk, ball, label
from skimage.filters import threshold_isodata, threshold_otsu
from collections import OrderedDict

def threshold_image(img, thresh, thresh_high_or_low='high'):
    if thresh_high_or_low == 'low':
        return img<thresh
    else:
        return img>thresh

def hysteresis_thresh(img, img_sample, infer=True, opening_ball_size=5, dilation_ball_size=2.5):
    if infer:
        thresh_otsu = threshold_otsu(img)
        low_thresh = (np.ceil(thresh_otsu/500)*500)-1000
        high_thresh = low_thresh+2000
        thresh_vals = [low_thresh, high_thresh]
        print(thresh_vals)
    else:
        thresh_vals = [0,20000]
        print(thresh_vals)
    
    step_list = OrderedDict()
    step_list['Strict Threshold']     = img>thresh_vals[0]
    step_list['Remove Small Objects'] = opening(step_list['Strict Threshold'], ball(opening_ball_size))
    step_list['Looser Threshold']     = img>thresh_vals[1]
    step_list['Both Thresholds']      = 1.0*step_list['Looser Threshold'] + 1.0*step_list['Remove Small Objects']

    # the tricky part keeping the between images
    step_list['Connected Thresholds'] = step_list['Remove Small Objects']

    for i in range(10):
        if i==5:
            print("Halfway there")
        step_list['Connected Thresholds'] = dilation(step_list['Connected Thresholds'] , 
                                                     ball(dilation_ball_size)) & step_list['Looser Threshold']

    fig, ax_steps = plt.subplots(1, len(step_list)+1, figsize = (18, 5), dpi = 150)

    for i, (c_ax, (c_title, c_img)) in enumerate(zip(ax_steps.flatten(), step_list.items()),1):
        c_ax.imshow(c_img[img_sample], cmap = 'gray' if c_img.max()<=1 else 'viridis')
        c_ax.set_title('%d) %s' % (i, c_title)); c_ax.axis('off');
        if i==5:
            connthresh_img = c_img

    ax_steps[5].imshow(img[img_sample], cmap='gray')
    ax_steps[5].set_title('%d) %s' % (6, "Input Image")); ax_steps[5].axis('off')
    
    return connthresh_img