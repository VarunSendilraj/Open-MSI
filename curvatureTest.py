import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
from skimage.filters import median

def visualize_elliptical_shapes(image_path, circle_size_threshold):
    # Load the BMP image using PIL
    image = Image.open(image_path)
    
    # Convert the image to grayscale
    image_gray = image.convert('L')
    
    # Convert the grayscale image to a NumPy array
    image_array = np.array(image_gray)
    
    # Apply heavy median filtering to the image
    filtered_image = median(image_array, selem=np.ones((5,5)))
    
    # Binarize the filtered image using a threshold
    threshold = 128  # Adjust the threshold value as needed
    binary_image = filtered_image > threshold
    
    # Label connected regions in the binary image
    labeled_image = label(binary_image)
    
    # Calculate the elliptical score for each labeled region
    elliptical_scores = []
    for region in regionprops(labeled_image):
        # Filter out small regions based on area
        if region.area >= circle_size_threshold:
            # Calculate the elliptical score as the ratio of the minor to major axis length
            elliptical_score = region.minor_axis_length / region.major_axis_length
            elliptical_scores.append(elliptical_score)
    
    # Create a binary image with only the labeled regions
    binary_image_labeled = labeled_image > 0
    
    # Display the original image, the image with median filtering applied, and the binary image
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(filtered_image, cmap='gray')
    axes[1].set_title('Filtered Image')
    axes[1].axis('off')
    axes[2].imshow(binary_image_labeled, cmap='gray')
    axes[2].set_title('Elliptical Shapes')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
image_path = 'MSI_image.bmp'
circle_size_threshold = 100  # Adjust the threshold value as needed
visualize_elliptical_shapes(image_path, circle_size_threshold)
