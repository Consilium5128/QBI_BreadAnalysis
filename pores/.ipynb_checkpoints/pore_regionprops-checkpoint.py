import numpy as np

def rough_porosity(merged_img, filled):
    size_merged_img = merged_img[filled].size
    pores_merged_img = np.sum((merged_img[filled]>0))
    print("Porosity (%) = "+str(100*pores_merged_img/size_merged_img))
    
def sphericity_pore(volume, surface_area):
    return ((np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area)