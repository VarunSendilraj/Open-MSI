import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Function to generate random data
def generate_data(num_points):
    np.random.seed(42)
    x = np.random.rand(num_points)
    y = np.random.rand(num_points)
    return x, y

# Function to perform k-means clustering
def perform_kmeans_clustering(x, y, num_clusters):
    # Perform k-means clustering here
    # Return the cluster labels

# Generate initial data and cluster labels
num_points = 100
x, y = generate_data(num_points)
num_clusters = 3
cluster_labels = perform_kmeans_clustering(x, y, num_clusters)

# Create the figure and plot the initial scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter(x, y, c=cluster_labels, cmap='viridis')

# Define the slider's position and range
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])  # Adjust the position and size as needed
slider = Slider(ax_slider, 'Number of Clusters', valmin=1, valmax=10, valinit=num_clusters)  # Adjust the range and initial value as needed

# Update function to be called when the slider's value changes
def update_clusters(val):
    num_clusters = int(slider.val)
    cluster_labels = perform_kmeans_clustering(x, y, num_clusters)
    scatter.set_array(cluster_labels)
    plt.draw()

slider.on_changed(update)

# Display the plot
plt.show()
