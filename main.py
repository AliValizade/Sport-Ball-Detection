import numpy as np
import cv2
from matplotlib import pyplot as plt

from modules.median import median_filter, adaptive_median_filter
from modules.gaussian import gaussian_filter, gaussian_kernel
from modules.hist import equalize_histogram, calc_hsv_histogram, dominant_color_analysis
from modules.restore import interpolate_missing_data
from modules.hough_circle import draw_circles, detect_circles


# Read image
img = cv2.imread('img/noisy-color(23).jpg')
if img is None:
    print(f"Error reading image: ")
    
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
img = cv2.resize(img, (256, 256))
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Convert RGB to GRAY

# Gaussian filter
kernel = gaussian_kernel(kernel_size=5, sigma=1.8)
gaussian_filtered_image = gaussian_filter(gray, kernel)

# Median filter
median_filtered_image = median_filter(gray, 5)

# Adaptive Median filter
adaptive_median_filtered_image = adaptive_median_filter(gray, 3)

# Increase contrast
contrast_image, _ = equalize_histogram(median_filtered_image)

# Convert to HSV
hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# Calculate color histogram
hist_h, hist_s, hist_v = calc_hsv_histogram(hsv_image)

# Dominant color analysis
color_features = dominant_color_analysis(hsv_image)
print(color_features)

# Image restoration (assumption: black areas have incomplete information)
image = cv2.cvtColor(contrast_image, cv2.COLOR_GRAY2RGB)
noisy_image = image.copy()
noisy_image[50:100, 50:100] = 0  # Create a region with incomplete data
restored_image = interpolate_missing_data(noisy_image)

# Using Hough Circle Transform to detect circles
circles = detect_circles(image)

# Draw circles on the image
image_with_circles = draw_circles(image, circles)

# Show images
plt.figure(figsize=(15, 10))
plt.subplot(3, 3, 1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(3, 3, 2)
plt.title("Gray Image")
plt.imshow(gray, cmap='gray')
plt.axis("off")

plt.subplot(3, 3, 3)
plt.title("Gaussian Filtered")
plt.imshow(gaussian_filtered_image, cmap='gray')
plt.axis("off")

plt.subplot(3, 3, 4)
plt.title("Median_filtered Image")
plt.imshow(median_filtered_image, cmap='gray')
plt.axis("off")

plt.subplot(3, 3, 5)
plt.title("Adaptive Median filtered Image")
plt.imshow(adaptive_median_filtered_image, cmap='gray')
plt.axis("off")

plt.subplot(3, 3, 6)
plt.title("Histogram Equalized (Contrast)")
plt.imshow(contrast_image, cmap='gray')
plt.axis("off")

plt.subplot(3, 3, 7)
plt.title("HSV Image")
plt.imshow(hsv_image)
plt.axis("off")

plt.subplot(3, 3, 8)
plt.title("Balls detected")
plt.imshow(image_with_circles)
plt.axis("off")

plt.tight_layout()
plt.show()

# Show Histogram
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Hue Histogram")
plt.plot(hist_h, color='red')
plt.xlabel("Bins")
plt.ylabel("Frequency")

plt.subplot(1, 3, 2)
plt.title("Saturation Histogram")
plt.plot(hist_s, color='green')
plt.xlabel("Bins")
plt.ylabel("Frequency")

plt.subplot(1, 3, 3)
plt.title("Value Histogram")
plt.plot(hist_v, color='blue')
plt.xlabel("Bins")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
