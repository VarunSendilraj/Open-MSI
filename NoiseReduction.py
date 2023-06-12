import numpy as np
import matplotlib.pyplot as plt

# Import the data
data = np.loadtxt("230111_caffeini_3_195_1.txt")

# Convert the data to a grid format
x = np.linspace(0, 1, data.shape[0])
y = np.linspace(0, 1, data.shape[1])
image = np.zeros((x.size, y.size))
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        image[i, j] = data[i, j]

# Perform image processing
# Background subtraction
background = np.mean(image)
image = image - background

# Flatten the image
image_flat = image.flatten()

# Smoothing
image_flat = np.convolve(image_flat, np.ones((5, 5), dtype=float), mode="same")

# Reshape the image
image = image_flat.reshape(x.size, y.size)

# Thresholding
threshold = np.mean(image) + 2 * np.std(image)
image[image < threshold] = 0

# Visualize the data
plt.imshow(image, cmap="gray")
plt.show()
