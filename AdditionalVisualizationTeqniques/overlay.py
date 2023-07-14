import cv2
import numpy as np
import matplotlib.pyplot as plt

# Assuming generated_image and ground_truth_image have the same dimensions

# Convert the generated_image to grayscale
generated_image = cv2.imread('MSI_image.bmp')
generated_image_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)

# Normalize the grayscale image to a range between 0 and 255
generated_image_gray_norm = cv2.normalize(generated_image_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# Create a 3-channel grayscale image from the normalized grayscale image
generated_image_rgb = generated_image

# Resize the ground_truth_image to match the dimensions of generated_image
ground_truth_image = cv2.imread('11 1.jpg')
ground_truth_resized = cv2.resize(ground_truth_image, (generated_image.shape[1], generated_image.shape[0]))

# Convert the ground_truth_image to grayscale
ground_truth_gray = cv2.cvtColor(ground_truth_resized, cv2.COLOR_BGR2GRAY)

# Convert the ground_truth_image to a 3-channel grayscale image
ground_truth_rgb = cv2.cvtColor(ground_truth_gray, cv2.COLOR_GRAY2BGR)

# Set the opacity of the ground_truth_rgb image (0.5 for 50% opacity, adjust as needed)
opacity = 0.5

# Overlay the ground_truth_rgb image onto the generated_image_rgb
overlay = cv2.addWeighted(ground_truth_rgb, 1 - opacity, generated_image_rgb, opacity, 0)

# Display the result
plt.imshow(overlay, cmap="jet")
plt.show()
