import numpy as np

def gaussian_kernel(kernel_size, sigma):
    # Create Gaussian Kernel
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def gaussian_filter(image, kernel):
    # Image dimensions and kernel
    height, width = image.shape
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    # Output image and padded image
    filtered_image = np.zeros_like(image, dtype=np.float32)
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    
    # Apply Gaussian filter
    for i in range(pad, height + pad):
        for j in range(pad, width + pad):
            # Extract window
            window = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            # Element-by-element multiplication and final sum
            filtered_image[i - pad, j - pad] = np.sum(window * kernel)
    
    return filtered_image.astype(np.uint8)
