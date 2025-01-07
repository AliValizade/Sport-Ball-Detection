import numpy as np

def median_filter(image, kernel_size):
    # Image dimensions
    height, width = image.shape
    # Initialize the output image to zero
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    # Filter window margin
    pad = kernel_size // 2
    # Pad the image with zero padding
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    
    # Apply median filter
    for i in range(pad, height + pad):
        for j in range(pad, width + pad):
            # Extract window (kernel)
            kernel = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            # Calculate the median value
            filtered_image[i - pad, j - pad] = np.median(kernel)
            
    
    return filtered_image



# Adaptive Median Filter (De-Noising)
def adaptive_median_filter(image, max_window_size=7):
    padded_image = np.pad(image, max_window_size // 2, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)
    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            window_size = 3
            while window_size <= max_window_size:
                half_size = window_size // 2
                window = padded_image[i:i + window_size, j:j + window_size]
                
                Z_min = np.min(window)
                Z_max = np.max(window)
                Z_med = np.median(window)
                Z_xy = padded_image[i + half_size, j + half_size]
                
                if Z_min < Z_med < Z_max:
                    if Z_min < Z_xy < Z_max:
                        filtered_image[i, j] = Z_xy
                    else:
                        filtered_image[i, j] = Z_med
                    break
                else:
                    window_size += 2
            
            if window_size > max_window_size:
                filtered_image[i, j] = Z_med

    return filtered_image

