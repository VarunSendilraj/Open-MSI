import struct
import cv2

def save_image_as_biomap(filename, image_data, num_channels, bit_depth, acquisition_params):
    # Header information
    image_size = len(image_data)
    
    # Write the header
    header = struct.pack('IIII', image_size, num_channels, bit_depth, acquisition_params)
    
    # Write the image data
    with open(filename, 'wb') as file:
        file.write(header)
        file.write(image_data)

# Example usage
image_data = cv2.imread('GuiCode\MSI_image.bmp')# Replace with actual image data
num_channels = 3
bit_depth = 8
acquisition_params = 123456789  # Replace with actual acquisition parameters

save_image_as_biomap('GuiCode\MSI_image.img', image_data, num_channels, bit_depth, acquisition_params)
