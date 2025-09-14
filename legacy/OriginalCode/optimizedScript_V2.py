import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import Image


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


# Mass Alignment between to provide accurate representation of chemical
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


# peak picking  --> Being able to increase the singal to noise ratio while preserving all important features
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


# implement the three steps of noise reduction here,


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
from matplotlib.colors import LinearSegmentedColormap


def transparent_colormap(existing_cmap_name = 'jet'):
    ncolors = 256
    color_array = plt.get_cmap(existing_cmap_name)(range(ncolors))

    # change alpha value of first color
    color_array[0,-1] = 0.0

    # create a colormap object
    new_cmap = LinearSegmentedColormap.from_list(name=f'{existing_cmap_name}_alpha', colors=color_array)

    # register this new colormap with matplotlib
    plt.register_cmap(cmap=new_cmap)
    
    return new_cmap


def save_image(reshaped_data, filename):
    """
    Save the mass spectrometry image as an image file.
    """
    image = np.array(reshaped_data)
    new_cmap = transparent_colormap()
    plt.imsave(filename, image, cmap=new_cmap)




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


def perform_kmeans_clustering(reshaped_data, num_clusters):
    """
    Perform k-means clustering on the reshaped data.
    """
    flattened_data = np.array(reshaped_data).flatten()
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(flattened_data.reshape(-1, 1))
    cluster_labels = kmeans.labels_
    return cluster_labels

def perform_pca(data):
    # Initialize PCA with desired number of components
    pca = PCA(n_components=2)
    
    # Perform PCA on the data
    pca_data = pca.fit_transform(data)
    
    return pca_data


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
    
    # Perform PCA on the scaled intensities
    #pca_data = perform_pca(np.resscaled_intensities)

    # Step 6: Reshape the data for visualization
    reshaped_data = reshape_data_for_visualization(scaled_intensities, Length)

    # Step 7: Reverse alternate rows
    reversed_data = reverse_alternate_rows(reshaped_data)


    # Perform k-means clustering on the reshaped data
    cluster_labels = perform_kmeans_clustering(reversed_data, 100)

    # Generate x and y coordinate arrays based on the shape of reversed_data
    x, y = np.meshgrid(np.arange(len(reversed_data[0])), np.arange(len(reversed_data)))

    # Plot K-means clustering results
    plt.scatter(x.flatten(), y.flatten(), c=cluster_labels, cmap='jet')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-means Clustering Results')
    plt.colorbar(label='Cluster Labels')
    plt.show()

    # Step 8: Save the mass spectrometry image
    save_image(reversed_data, 'MSI_image.bmp')

    # Step 9: Display the mass spectrometry image with colorbar
    display_image_with_colorbar(reversed_data)


if __name__ == '__main__':
    main()
