import numpy as np
import cv2

def calculate_histogram(image):
    # Calculate the image histogram
    hist = [0] * 256
    for row in image:
        for pixel in row:
            hist[pixel] += 1
    return hist

def calculate_cumulative_histogram(hist):
    # Calculate the cumulative histogram
    cum_hist = [0] * 256
    cum_hist[0] = hist[0]
    for i in range(1, 256):
        cum_hist[i] = cum_hist[i - 1] + hist[i]
    return cum_hist

def equalize_histogram(image):
    # Histogram equalization
    hist = calculate_histogram(image)
    cum_hist = calculate_cumulative_histogram(hist)
    total_pixels = image.size
    normalized_cum_hist = [ch / total_pixels for ch in cum_hist]
    equalized_map = [int(nch * 255) for nch in normalized_cum_hist]

    # Apply new values ​​to the image
    equalized_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized_image[i, j] = equalized_map[image[i, j]]

    return equalized_image, hist


# Extract color histogram
def calc_hsv_histogram(image, bins=256):
    hist_h = cv2.calcHist([image], [0], None, [bins], [0, 256])  # Hue Channel
    hist_s = cv2.calcHist([image], [1], None, [bins], [0, 256])  # Saturation Channel
    hist_v = cv2.calcHist([image], [2], None, [bins], [0, 256])  # Value Channel
    return hist_h, hist_s, hist_v


# Dominant color analysis
def dominant_color_analysis(image):
    # Separate channels
    h, s, v = cv2.split(image)
    
    # Mean and standard deviation of each channel
    mean_h = np.mean(h)
    std_h = np.std(h)
    mean_s = np.mean(s)
    std_s = np.std(s)
    mean_v = np.mean(v)
    std_v = np.std(v)
    
    return {
        "Mean_H": mean_h, "Std_H": std_h,
        "Mean_S": mean_s, "Std_S": std_s,
        "Mean_V": mean_v, "Std_V": std_v,
    }