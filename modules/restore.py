import numpy as np

# Image restoration with simple interpolation
def interpolate_missing_data(image):
    restored_image = image.copy()
    rows, cols, channels = image.shape
    
    for c in range(channels):
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if restored_image[i, j, c] == 0:  # Assume pixel zero is defective
                    neighbors = [
                        restored_image[i - 1, j, c], restored_image[i + 1, j, c],
                        restored_image[i, j - 1, c], restored_image[i, j + 1, c]
                    ]
                    restored_image[i, j, c] = int(np.mean([n for n in neighbors if n > 0]))
    
    return restored_image
