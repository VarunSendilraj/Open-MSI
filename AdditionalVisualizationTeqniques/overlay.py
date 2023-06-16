import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the two input images
overlay_image = cv2.imread('GuiCode\MSI_image.bmp')

background_image= cv2.imread('11 1.jpg')

# Resize images to have the same dimensions
height = min(background_image.shape[0], overlay_image.shape[0])
width = min(background_image.shape[1], overlay_image.shape[1])
background_image = cv2.resize(background_image, (width, height))
overlay_image = cv2.resize(overlay_image, (width, height))

# Convert images to grayscale
background_gray = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
overlay_gray = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2GRAY)

# Apply edge detection to grayscale images
background_edges = cv2.Canny(background_gray, 100, 200)
overlay_edges = cv2.Canny(overlay_gray, 100, 200)

# Create a mask by thresholding the edge images
_, background_mask = cv2.threshold(background_edges, 100, 255, cv2.THRESH_BINARY)
_, overlay_mask = cv2.threshold(overlay_edges, 100, 255, cv2.THRESH_BINARY)

# Dilate the mask to ensure a smoother transition
kernel = np.ones((5, 5), np.uint8)
background_mask = cv2.dilate(background_mask, kernel, iterations=1)
overlay_mask = cv2.dilate(overlay_mask, kernel, iterations=1)

# Create copies of the original images
background_copy = background_image.copy()
overlay_copy = overlay_image.copy()

# Apply the mask to the overlay image
overlay_copy[overlay_mask != 0] = 0

# Add the masked overlay image to the background image
result = cv2.add(background_copy, overlay_copy)

# Display the result
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
