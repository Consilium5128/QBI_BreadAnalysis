import numpy as np

def crop_image_square_3d(img, bound1, bound2):
    return img[:,bound1:bound2,bound1:bound2]

def crop_image_square_2d(img, bound1, bound2):
    return img[bound1:bound2, bound1:bound2]

def downsampling_3d(img, downsampling_factor):
    return img[:,::downsampling_factor,::downsampling_factor]

def downsampling_2d(img, downsampling_factor):
    return img[::downsampling_factor,::downsampling_factor]