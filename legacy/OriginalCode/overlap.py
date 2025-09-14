import cv2
import numpy as np

def overlay_images(microscope_img, intensity_img, colormap=cv2.COLORMAP_JET):
    # Convert intensity image to colormap
    intensity_colormap = cv2.applyColorMap(intensity_img, colormap)
    
    # Preprocess the images (resize if necessary)
    if microscope_img.shape != intensity_colormap.shape:
        intensity_colormap = cv2.resize(intensity_colormap, (microscope_img.shape[1], microscope_img.shape[0]))
    
    # Convert images to grayscale
    microscope_gray = cv2.cvtColor(microscope_img, cv2.COLOR_BGR2GRAY)
    intensity_gray = cv2.cvtColor(intensity_colormap, cv2.COLOR_BGR2GRAY)
    
    # Perform image registration
    warp_mode = cv2.MOTION_TRANSLATION  # Change if needed (e.g., cv2.MOTION_EUCLIDEAN)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
    _, warp_matrix = cv2.findTransformECC(intensity_gray, microscope_gray, warp_matrix, warp_mode, criteria)
    
    # Warp the intensity image
    aligned_intensity = cv2.warpAffine(intensity_colormap, warp_matrix, (microscope_img.shape[1], microscope_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    # Overlay the aligned intensity image on the microscope image
    overlay = cv2.addWeighted(microscope_img, 0.7, aligned_intensity, 0.3, 0)
    
    return overlay

# Example usage
microscope_img = cv2.imread('11 1.jpg')
intensity_img = cv2.imread('guiSavedImages\caffineSample.bmp', cv2.IMREAD_GRAYSCALE)

overlay = overlay_images(microscope_img, intensity_img)

# Display the overlay image
cv2.imshow('Overlay', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
