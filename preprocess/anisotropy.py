import numpy as np
from scipy.fftpack import fftn, fftshift
from skimage.feature import structure_tensor, structure_tensor_eigenvalues

def infer_anisotropy_fft_3d(img):
    ft_image = fftshift(fftn(img))
    power_spectrum = np.abs(ft_image)**2

    # Calculate the mean power spectrum along each axis
    mean_power_spectrum = [np.mean(power_spectrum, axis=i) for i in range(3)]

    # Identify anisotropy by comparing the power spectrum in different directions
    anisotropy_ratios = [mean_power_spectrum[i].max() / mean_power_spectrum[i].min() for i in range(3)]
    print(f"Anisotropy Ratios: {anisotropy_ratios}")

    # Normalize the anisotropy ratios to get the scaling factors
    scaling_factors = [1.0 / ratio for ratio in anisotropy_ratios]
    #print(f"Scaling Factors: {scaling_factors}")
    
    return scaling_factors

def infer_anisotropy_structure_tensors_3d(img):
    # Calculate the structure tensor of the image stack
    A_elems = structure_tensor(img, sigma=1, order='rc')

    # Calculate eigenvalues of the structure tensor
    eigenvalues = structure_tensor_eigenvalues(A_elems)

    # The eigenvalues can give us an idea of the anisotropy
    # We will calculate the average eigenvalues in each direction
    mean_eigenvalues = [np.mean(eigenvalues[i]) for i in range(3)]

    print(f"Mean eigenvalues: {mean_eigenvalues}")

    # Normalize the eigenvalues to get the scaling factors
    scaling_factors = [mean_eigenvalues[0] / min(mean_eigenvalues),
                       mean_eigenvalues[1] / min(mean_eigenvalues),
                       mean_eigenvalues[2] / min(mean_eigenvalues)]

    #print(f"Scaling factors: {scaling_factors}")
    return scaling_factors