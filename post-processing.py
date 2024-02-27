import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

def max_cross_correlation(image1, image2):

    # Compute cross-correlation
    cross_corr = correlate2d(image1, image2, mode='same')

    # Find the indices of maximum value in the cross-correlation matrix
    max_corr_index = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
    max_corr_value = cross_corr[max_corr_index]

    return max_corr_index, max_corr_value, cross_corr

def plot_cross_correlation(image1, image2, cross_corr, FileName):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image1, cmap='gray')
    axes[0].set_title('Image 1')

    axes[1].imshow(image2, cmap='gray')
    axes[1].set_title('Image 2')

    axes[2].imshow(cross_corr, cmap='viridis')
    axes[2].set_title('Cross-correlation')
    axes[2].plot(max_corr_index[1], max_corr_index[0], 'ro')

    #plot colorbar
    cax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(axes[2].imshow(cross_corr, cmap='viridis'), cax=cax, ax=axes[2])
    cbar.set_label('Cross-correlation')
    plt.savefig(FileName + '.png')
    plt.show()

# Load images as numpy arrays from text files with index 0 and 1
Image0In =  np.loadtxt('./DATA/0_input.txt')
Image0Out = np.loadtxt('./DATA/0_output.txt')
Image1In =  np.loadtxt('./DATA/1_input.txt')
Image1Out = np.loadtxt('./DATA/1_output.txt')

# Compute maximum cross-correlation
max_corr_index, max_corr_value, cross_corr = max_cross_correlation(Image0In, Image0Out)
print(f"Maximum cross-correlation value for 0: {max_corr_value}")
print(f"Location of maximum correlation for 1: {max_corr_index}")
max_corr_index, max_corr_value, cross_corr = max_cross_correlation(Image1In, Image1Out)
print(f"Maximum cross-correlation value for 1: {max_corr_value}")
print(f"Location of maximum correlation for 1: {max_corr_index}")

# Plot the images and cross-correlation with colorbar
plot_cross_correlation(Image0In, Image0Out, cross_corr, './DATA/0_cross_correlation')
plot_cross_correlation(Image1In, Image1Out, cross_corr, './DATA/1_cross_correlation')
