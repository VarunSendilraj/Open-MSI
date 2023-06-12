import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def read_data_file(filename):
    """
    Read the data file and split it by newline.
    """
    with open(filename, 'r') as file:
        data = file.read().split("\n")
    return data


def preprocess_data(data):
    """
    Preprocess the data by replacing tabs with commas and converting to floats.
    """
    preprocessed_data = []
    for line in data:
        line = line.replace('\t', ',')
        line = line.split(',')
        line = [float(x) for x in line]
        preprocessed_data.append(line)
    return preprocessed_data

#Mass Alignment between to provide accurate representation of chemical 
def collect_data_by_ablation_time(preprocessed_data, start_time, ablation_time):
    """
    Collect data for each ablation time interval.
    """
    collected_data = []
    current_time = start_time
    current_interval_data = []
    
    for line in preprocessed_data:
        if line[0] < current_time:
            continue
        elif current_time <= line[0] < current_time + ablation_time:
            current_interval_data.append(line)
        else:
            current_interval_data.append(line)
            collected_data.append(current_interval_data)
            current_time += ablation_time
            current_interval_data = []
    
    return collected_data

#peak picking  --> Being able to increase the singal to noise ratio while preserving all improtatnt features
def subtract_background_noise(collected_data, noise):
    """
    Subtract background noise and calculate the sum of intensities for each interval.
    """
    intensities = []
    interval_intensity = 0
    
    for interval in collected_data:
        for line in interval:
            intensity = max(line[1] - noise, 0)
            interval_intensity += intensity
        intensities.append(interval_intensity)
        interval_intensity = 0
    
    return intensities

#implement the threee steps of noise reduction here, 

def scale_and_log_transform(intensities, ablation_times):
    """
    Scale and perform logarithmic transformation on the intensities.
    """
    scaled_intensities = np.zeros(ablation_times)
    
    for i in range(min(len(scaled_intensities), len(intensities))):
        if intensities[i] > 0:
            scaled_intensities[i] = math.log(intensities[i], 10)
    
    return scaled_intensities


def reshape_data_for_visualization(scaled_intensities, length):
    """
    Reshape the data for image visualization.
    """
    reshaped_data = []
    
    for i in range(0, len(scaled_intensities), length):
        reshaped_data.append(scaled_intensities[i:i + length])
    
    return reshaped_data


def reverse_alternate_rows(reshaped_data):
    """
    Reverse alternate rows in the reshaped data.
    """
    reversed_data = []
    
    for i in range(len(reshaped_data)):
        if i % 2 == 0:
            reversed_data.append(reshaped_data[i])
        else:
            reversed_data.append(reshaped_data[i][::-1])
    
    return reversed_data




def save_image(reshaped_data, filename):
    """
    Save the mass spectrometry image as an image file.
    """
    image = np.array(reshaped_data)
    #everything below until plt.imsave is code to save it as a .img file
    #image_array = image.astype(np.uint8)

    # Save the image array as a binary file with .img extension
    #with open(filename, 'wb') as file:
    #    file.write(image_array.tobytes())
    
    plt.imsave(filename, image, cmap="jet")


def display_image_with_colorbar(reshaped_data):
    """
    Display the mass spectrometry image with a colorbar.
    """
    image = np.array(reshaped_data)
    plt.imshow(image, cmap="jet")
    plt.colorbar(label='log(Signal intensity [arb. units])')
    plt.xticks([0, 25, 50, 75, 100], [0, 1, 2, 3, "4\nmm"], fontsize=20)
    plt.yticks([0, 25, 50, 75, 100], ["4\nmm", 3, 2, 1, 0], fontsize=20)
    plt.show()


def main():
    # Specify input data and parameters
    IMS_data = 'ToFData\\230111_caffeini_3_195_1.txt'
    AbrationTime = 0.05054
    StartTime = 1.0
    Noise = 50
    Length = 100
    AbrationTimes = 10000

    # Step 1: Read the data file
    data = read_data_file(IMS_data)

    # Step 2: Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Step 3: Collect data for each ablation time interval
    collected_data = collect_data_by_ablation_time(preprocessed_data, StartTime, AbrationTime)

    # Step 4: Subtract background noise and calculate intensities
    intensities = subtract_background_noise(collected_data, Noise)

    # Step 5: Scale and perform logarithmic transformation
    scaled_intensities = scale_and_log_transform(intensities, AbrationTimes)

    # Step 6: Reshape the data for visualization
    reshaped_data = reshape_data_for_visualization(scaled_intensities, Length)

    # Step 7: Reverse alternate rows
    reversed_data = reverse_alternate_rows(reshaped_data)

    # Step 8: Save the mass spectrometry image
    save_image(reversed_data, 'MSI_image.bmp')

    # Step 9: Display the mass spectrometry image with colorbar
    display_image_with_colorbar(reversed_data)


if __name__ == '__main__':
    main()
