import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import subprocess
import os
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.cm import get_cmap
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



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
    # Normalize the intensities
    max_intensity = max(intensities)
    normalized_intensities = [intensity / max_intensity for intensity in intensities]

    return intensities


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


def save_image(reshaped_data, filename,colormap):
    """
    Save the mass spectrometry image as an image file.
    """
    image = np.array(reshaped_data)
    plt.imsave(filename, image, cmap=colormap)
    
def save_image_as_img(reshaped_data, filename):
    """
    Save the mass spectrometry image as a .img file.
    """
    # Convert the reshaped_data to a 2D array and scale it to the range [0, 255]
    img_data = np.array(reshaped_data)
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255

    # Convert the image data to uint8
    img_data = img_data.astype(np.uint8)

    # Save the image data as a binary file
    img_data.tofile(filename)

    print(f"Image saved as {filename}")

import matplotlib.colors as colors
import matplotlib.cm as cm

from matplotlib.colors import LinearSegmentedColormap

'''
def transparent_colormap(existing_cmap_name: str):
    ncolors = 256
    color_array = plt.get_cmap(existing_cmap_name)(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.linspace(1.0, 0.0, ncolors)

    # create a colormap object
    new_cmap = LinearSegmentedColormap.from_list(name=f'{existing_cmap_name}_alpha', colors=color_array)

    # register this new colormap with matplotlib
    plt.register_cmap(cmap=new_cmap)
    
    return new_cmap


def save_image_clear(reshaped_data, filename,colormap):
    """
    Save the mass spectrometry image as an image file.
    """
    image = np.array(reshaped_data)
    new_cmap = transparent_colormap(colormap)
    plt.imsave(filename, image, cmap=new_cmap)

'''
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
    flattened_data = np.array(reshaped_data).flatten()
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(flattened_data.reshape(-1, 1))
    cluster_labels = kmeans.labels_.reshape(np.array(reshaped_data).shape)
    return cluster_labels





def find_voxel_value_range(image_volume):
    # Initialize max and min values
    max_value = float('-inf')
    min_value = float('inf')

    # Iterate over each voxel in the image volume
    for z in range(image_volume.shape[0]):
        for y in range(image_volume.shape[1]):
            for x in range(image_volume.shape[2]):
                voxel_value = image_volume[z, y, x]
                if voxel_value > max_value:
                    max_value = voxel_value
                if voxel_value < min_value:
                    min_value = voxel_value

    return max_value, min_value

# Define a function to create the mass spectrometry image using Plotly
def create_mass_spectrometry_image(reshaped_data):
    fig = make_subplots(rows=1, cols=1)
    fig.add_heatmap(z=reshaped_data, colorscale="jet")
    fig.update_layout(
        title='Mass Spectrometry Image',
        xaxis_title='X',
        yaxis_title='Y'
    )
    fig.show()



def main():
    def select_image(button, variable):
        image_path = filedialog.askopenfilename(title="Select Image")
        if image_path:
            button["text"] = "Selected: " + image_path
            variable.set(image_path)

    def start_application():
        initial.destroy()

    # Create the initial tkinter window
    initial = tk.Tk()
    initial.title("Start Screen")

    # Load and display a logo image
    logo_image = Image.open("main.ico")  # Replace "logo.png" with the path to your logo image
    logo_image = logo_image.resize((200, 200))  # Resize the image as per your requirements
    logo_photo = ImageTk.PhotoImage(logo_image)

    logo_label = tk.Label(initial, image=logo_photo)
    logo_label.pack(pady=10)

    # Add text labels
    title_label = tk.Label(initial, text="Welcome to MSI Generator", font=("Helvetica", 16, "bold"))
    title_label.pack(pady=10)

    instructions_label = tk.Label(initial, text="Please select an refrence image and time of flight data.", font=("Helvetica", 12))
    instructions_label.pack(pady=5)

    # Create StringVar variables for the file paths
    microscope_image_path = tk.StringVar()
    tof_data_path = tk.StringVar()

    # Create buttons for image and data selection
    microscope_image_button = ttk.Button(initial, text="Select Microscope Image",
                                        command=lambda: select_image(microscope_image_button, microscope_image_path))
    microscope_image_button.pack(pady=10)

    tof_data_button = ttk.Button(initial, text="Select Time of Flight Data",
                                command=lambda: select_image(tof_data_button, tof_data_path))
    tof_data_button.pack(pady=10)

    # Create the Start button
    start_button = ttk.Button(initial, text="Start", command=start_application)
    start_button.pack(pady=20)

    # Set the window dimensions and center it on the screen
    window_width = 400
    window_height = 500

    screen_width = initial.winfo_screenwidth()
    screen_height = initial.winfo_screenheight()

    x = int((screen_width / 2) - (window_width / 2))
    y = int((screen_height / 2) - (window_height / 2))

    initial.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Run the tkinter event loop
    initial.mainloop()

    # Access the selected file paths
    microscope_image_file_path = microscope_image_path.get()
    
    tof_data_file_path = tof_data_path.get()

    # Check if both paths are selected
    if microscope_image_file_path and tof_data_file_path:
        print("Microscope Image Path:", microscope_image_file_path)
        print("Time of Flight Data Path:", tof_data_file_path)
        # Further processing with the file paths can be done here
    else:
        print("File paths are not selected.")
    
    
    root = tk.Tk()
    root.withdraw()
    IMS_data = tof_data_file_path
    root.destroy()
    

    if not IMS_data:
        return

    # Specify input data and parameters
    AbrationTime = 0
    StartTime = 0
    Noise = 0
    Length = 100
    AbrationTimes = 10000

    # Step 1: Read the data file
    def read_data_file(filename):
        with open(filename, 'r') as file:
            data = file.read().split("\n")
        return data

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

    #No need to save or display images
    # Step 8: Save the mass spectrometry image
    #save_image(reversed_data, 'MSI_image.bmp')

    # Step 9: Display the mass spectrometry image with colorbar
    #display_image_with_colorbar(reversed_data)

    # GUI code
    
    root = tk.Tk()
    root.title("Mass Spectrometry Image GUI")

    # Load and display the initial image
    image = Image.new('RGB', (400, 400), color='white')
    image = image.resize((400, 400), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(image)
    image_label = tk.Label(root, image=img_tk)
    image_label.pack(pady=10)


    def update_image(event=None):
        """
        Update the image based on the selected method for updating the ablation time.
        """
        use_entry_box = use_entry_box_var.get()
        if use_entry_box:
            ablation_time_entry_value = ablation_entry.get()
            if ablation_time_entry_value:
                try:
                    ablation_time = float(ablation_time_entry_value)
                    ablation_time_slider.set(ablation_time)  # Set the slider value
                except ValueError:
                    print("Invalid ablation time value.")
                    return
        else:
            ablation_time = ablation_time_slider.get()
            ablation_entry.delete(0, tk.END)
            ablation_entry.insert(0, str(ablation_time))

        noise_value = noise_slider.get()
        start_time = float(start_time_entry.get())
        colormap = colormap_var.get()

        # Step 3: Collect data for each ablation time interval
        collected_data = collect_data_by_ablation_time(preprocessed_data, start_time, ablation_time)

        # Step 4: Subtract background noise and calculate intensities
        normalized_intensities = subtract_background_noise(collected_data, noise_value)

        # Step 5: Scale and perform logarithmic transformation
        scaled_intensities = scale_and_log_transform(normalized_intensities, AbrationTimes)

        # Step 6: Reshape the data for visualization
        reshaped_data = reshape_data_for_visualization(scaled_intensities, Length)

        # Step 7: Reverse alternate rows
        reversed_data = reverse_alternate_rows(reshaped_data)

        # Step 8: Save the mass spectrometry image
        save_image(reversed_data, 'MSI_image.bmp',colormap)
        
        

        # Update the image in the GUI
        new_image = Image.open('MSI_image.bmp')
        new_image = new_image.resize((400, 400), Image.LANCZOS)
        new_img_tk = ImageTk.PhotoImage(new_image)
        image_label.configure(image=new_img_tk)
        image_label.image = new_img_tk

        # Update the label text
        label_text.set(f"Ablation Time: {ablation_time:.5f} s  |  Noise: {noise_value}  |  Start Time: {start_time}")
    

    
    # Slider to adjust the ablation time
    ablation_time_label = tk.Label(root, text="Ablation Time (Mass Alignment):")
    ablation_time_label.pack(pady=5)
    ablation_time_slider = tk.Scale(root, from_=0.0450, to=0.0550, resolution=0.0000001, length=350,
                                    orient=tk.HORIZONTAL, command=update_image)
    ablation_time_slider.set(AbrationTime)
    ablation_time_slider.pack()
    
    # Checkbutton for selecting the method to update the ablation time
    use_entry_box_var = tk.IntVar()
    use_entry_box_checkbutton = tk.Checkbutton(root, text="Use Entry Box (more percision):", variable=use_entry_box_var, command=update_image)
    use_entry_box_checkbutton.pack(pady=5)
    # Entry box for entering the ablation time
    #ablation_label = tk.Label(root, text="")
    #ablation_label.pack(pady=10)
    ablation_entry = tk.Entry(root)
    ablation_entry.insert(0, str(AbrationTime))
    ablation_entry.bind("<Return>", update_image)
    ablation_entry.pack()
    
    # Slider for adjusting the noise
    noise_label = tk.Label(root, text="Noise Reduction:")
    noise_label.pack()
    noise_slider = tk.Scale(root, from_=0, to=100, length=350, orient=tk.HORIZONTAL, resolution=1, command=update_image)
    noise_slider.pack()
     # Entry box for entering the start time
    start_time_label = tk.Label(root, text="Start Time (min):")
    start_time_label.pack(pady=10)
    start_time_entry = tk.Entry(root)
    start_time_entry.insert(0, str(StartTime))
    start_time_entry.bind("<Return>", update_image)
    start_time_entry.pack(pady=(0,15))
    
    colormap_frame = tk.Frame(root)
    colormap_frame.pack(pady=10)

    colormap_label = tk.Label(colormap_frame, text="Choose Colormap:")
    colormap_label.grid(row=0, column=0, padx=10)

    colormap_options = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'Blues', 'Reds', 'jet']  # Add more colormaps as needed
    colormap_var = tk.StringVar(root)
    colormap_var.set(colormap_options[0])  # Set the initial colormap

    colormap_menu = tk.OptionMenu(colormap_frame, colormap_var, *colormap_options, command=update_image)
    colormap_menu.grid(row=0, column=1, padx=10)
    
  
    
    # Save Image button
    def save_image_handler():
        ablation_time_entry_value = ablation_entry.get()
        if ablation_time_entry_value:
            try:
                ablation_time = float(ablation_time_entry_value)
            except ValueError:
                print("Invalid ablation time value.")
                return
        else:
            ablation_time = ablation_time_slider.get()
        noise_value = noise_slider.get()
        start_time = float(start_time_entry.get())
        colormap = colormap_var.get()


        # Open file dialog to choose save location and filename
        file_path = filedialog.asksaveasfilename(defaultextension=".bmp", filetypes=(("BMP files", "*.bmp"), ("All files", "*.*")))
        
        # Check if the user canceled the save operation
        if not file_path:
            return
        
        # Generate the filename based on the selected ablation time, noise, and start time
        data = read_data_file(IMS_data)

        # Step 2: Preprocess the data
        preprocessed_data = preprocess_data(data)

        # Step 3: Collect data for each ablation time interval
        collected_data = collect_data_by_ablation_time(preprocessed_data, start_time, ablation_time)

        # Step 4: Subtract background noise and calculate intensities
        intensities = subtract_background_noise(collected_data, noise_value)

        # Step 5: Scale and perform logarithmic transformation
        scaled_intensities = scale_and_log_transform(intensities, AbrationTimes)

        # Step 6: Reshape the data for visualization
        reshaped_data = reshape_data_for_visualization(scaled_intensities, Length)

        # Step 7: Reverse alternate rows
        reversed_data = reverse_alternate_rows(reshaped_data)

        save_image(reversed_data, file_path, colormap)
        print(f"Image saved as {file_path}")
    
    
        
    
    def save_img_handler2():
        ablation_time_entry_value = ablation_entry.get()
        if ablation_time_entry_value:
            try:
                ablation_time = float(ablation_time_entry_value)
            except ValueError:
                print("Invalid ablation time value.")
                return
        else:
            ablation_time = ablation_time_slider.get()
        noise_value = noise_slider.get()
        start_time = float(start_time_entry.get())
        colormap = colormap_var.get()
        # Open file dialog to choose save location and filename
        file_path = filedialog.asksaveasfilename(defaultextension=".img", filetypes=(("Image files", "*.img"), ("All files", "*.*")))
        
        # Check if the user canceled the save operation
        if not file_path:
            return

        # Generate the filename based on the selected ablation time, noise, and start time
        data = read_data_file(IMS_data)

        # Step 2: Preprocess the data
        preprocessed_data = preprocess_data(data)

        # Step 3: Collect data for each ablation time interval
        collected_data = collect_data_by_ablation_time(preprocessed_data, start_time, ablation_time)

        # Step 4: Subtract background noise and calculate intensities
        intensities = subtract_background_noise(collected_data, noise_value)

        # Step 5: Scale and perform logarithmic transformation
        scaled_intensities = scale_and_log_transform(intensities, AbrationTimes)

        # Step 6: Reshape the data for visualization
        reshaped_data = reshape_data_for_visualization(scaled_intensities, Length)

        # Step 7: Reverse alternate rows
        reversed_data = reverse_alternate_rows(reshaped_data)        
        save_image(reversed_data,'USE.bmp',colormap)
        useIMG = cv2.imread('USE.bmp')
        
        print(useIMG.shape)
        # Create a folder to hold the image files
        folder_name = f'{file_path[:-4]}-Biomap'  # Specify the folder name
        os.makedirs(folder_name, exist_ok=True)  # Create the folder if it doesn't exist
        img_file = os.path.join(folder_name, os.path.basename(file_path))
        print(folder_name)
        print(img_file)
    

        save_image_as_img(reversed_data, img_file)
        # Retrieve the values from GUI inputs
        data_types = ["CHAR", "BINARY", "SHORT", "INT", "FLOAT", "COMPLEX", "DOUBLE", "RGB" ]  #Available Datatypes
        input_window = tk.Toplevel(root)
        

        def save_image_with_input():
            # Retrieve the values from the input fields
            fileMain = os.path.basename(file_path)
            header_file = os.path.join(folder_name, f'{fileMain[:-4]}.hdr')
            print(folder_name)
            print(f'{file_path[:-4]}.hdr')
            print(f'Header File: {header_file}')
            width = int(width_entry.get())
            height = int(height_entry.get())
            depth = int(depth_entry.get())
            num_volumes = int(num_volumes_entry.get())
            data_type = data_type_var.get()
            max_value = int(max_value_entry.get())
            min_value = int(min_value_entry.get())

            # Call the compiled C program using subprocess.run()
            subprocess.run(["fileFormatting\make_hdr", header_file, str(width), str(height), str(depth),
                            str(num_volumes), data_type, str(max_value), str(min_value)])

            # Close the input window
            input_window.destroy()
        

        height, width,depth = useIMG.shape
        tk.Label(input_window, text="Width:").grid(row=1, column=0)
        width_entry = tk.Entry(input_window)
        width_entry.insert(0,width)
        width_entry.grid(row=1, column=1)

        tk.Label(input_window, text="Height:").grid(row=2, column=0)
        height_entry = tk.Entry(input_window)
        height_entry.insert(0,height)
        height_entry.grid(row=2, column=1)

        tk.Label(input_window, text="Depth:").grid(row=3, column=0)
        depth_entry = tk.Entry(input_window)
        depth_entry.insert(0,depth)
        depth_entry.grid(row=3, column=1)

        #only can be determined using a .dcm file (medical image file)
        tk.Label(input_window, text="Number of Volumes:").grid(row=4, column=0)
        num_volumes_entry = tk.Entry(input_window)
        num_volumes_entry.insert(0,depth)
        num_volumes_entry.grid(row=4, column=1)
       
        
        
        tk.Label(input_window, text="Data Type:").grid(row=5, column=0)
        data_type_var = tk.StringVar(input_window)
        data_type_var.set(data_types[4])  # Set default value
        data_type_menu = tk.OptionMenu(input_window, data_type_var, *data_types)
        data_type_menu.grid(row=5, column=1)

        maxval,minval = find_voxel_value_range(useIMG)
        tk.Label(input_window, text="Max Value:").grid(row=6, column=0)
        max_value_entry = tk.Entry(input_window)
        max_value_entry.insert(0, str(maxval))
        max_value_entry.grid(row=6, column=1)

        tk.Label(input_window, text="Min Value:").grid(row=7, column=0)
        min_value_entry = tk.Entry(input_window)
        min_value_entry.insert(0,str(minval))
        min_value_entry.grid(row=7, column=1)

        # Button to save the image with the entered values
        save_button = tk.Button(input_window, text="Save", command=save_image_with_input)
        save_button.grid(row=8, columnspan=2, pady=10)

    import mplcursors

    def further_analysis():
        ablation_time_entry_value = ablation_entry.get()
        if ablation_time_entry_value:
            try:
                ablation_time = float(ablation_time_entry_value)
            except ValueError:
                print("Invalid ablation time value.")
                return
        else:
            ablation_time = ablation_time_slider.get()
        noise_value = noise_slider.get()
        start_time = float(start_time_entry.get())
        colormap = colormap_var.get()
        # Generate the filename based on the selected ablation time, noise, and start time
        data = read_data_file(IMS_data)

        # Step 2: Preprocess the data
        preprocessed_data = preprocess_data(data)

        # Step 3: Collect data for each ablation time interval
        collected_data = collect_data_by_ablation_time(preprocessed_data, start_time, ablation_time)

        # Step 4: Subtract background noise and calculate intensities
        intensities = subtract_background_noise(collected_data, noise_value)

        # Step 5: Scale and perform logarithmic transformation
        scaled_intensities = scale_and_log_transform(intensities, AbrationTimes)

        # Step 6: Reshape the data for visualization
        reshaped_data = reshape_data_for_visualization(scaled_intensities, Length)

        # Step 7: Reverse alternate rows
        reversed_data = reverse_alternate_rows(reshaped_data)
        
        # Perform k-means clustering on the reshaped data
        cluster_labels = perform_kmeans_clustering(reversed_data, num_clusters=3)  # Adjust the number of clusters as needed
        from matplotlib.widgets import Slider


        # Convert the reversed_data list to a NumPy array
        image = np.array(reversed_data)
        image = image.astype(np.float32)

        
        
        
        # Create a subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Adjust the figsize as needed

        # Plot the main image
        ax1.imshow(image, cmap=colormap)
        ax1.set_title('Mass Spectrometry Image')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        # Create a cursor that displays the z-value when hovering over pixels
        cursor = mplcursors.cursor(hover=True)

        @cursor.connect("add")
        def on_add(sel):
            x, y = sel.target.index
            z = image[y, x]
            sel.annotation.set_text(f'Z: {z:.2f}')


                # Determine the minimum and maximum cluster labels
        min_label = np.min(cluster_labels)
        max_label = np.max(cluster_labels)

        # Plot the k-means clustering results with defined vmin and vmax
        x, y = np.meshgrid(np.arange(len(cluster_labels[0])), np.arange(len(cluster_labels)))
        scatter = ax2.scatter(x, len(cluster_labels) - y - 1, c=cluster_labels, cmap= colormap, vmin=min_label, vmax=max_label)

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('K-means Clustering Results')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Cluster Labels')

        # Adjust the aspect ratio
        ax2.set_aspect('equal')

        # Hide ticks and labels of subplot 2
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Define the slider's position and range
        slider_pos = inset_axes(ax2, width="75%", height="5%", loc='lower center', bbox_to_anchor=(0, -0.15, 1, 1), bbox_transform=ax2.transAxes)
        slider = Slider(slider_pos, 'K: ', valmin=1, valmax=20, valinit=10, valstep=1)  # Adjust the range and initial value as needed

        
        
        # Update function to be called when the slider's value changes
        def update_clusters(val):
            num_clusters = int(slider.val)
            cluster_labels = perform_kmeans_clustering(reversed_data, num_clusters)
            scatter.set_array(cluster_labels.flatten())

            # Update the colormap limits based on the new cluster labels
            scatter.set_clim(vmin=np.min(cluster_labels), vmax=np.max(cluster_labels))
            plt.draw()

        slider.on_changed(update_clusters)

        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.2)

        # Display the plot
        plt.show()




 
            
    def generate_3d_plot():
        ablation_time_entry_value = ablation_entry.get()
        if ablation_time_entry_value:
            try:
                ablation_time = float(ablation_time_entry_value)
            except ValueError:
                print("Invalid ablation time value.")
                return
        else:
            ablation_time = ablation_time_slider.get()
        noise_value = noise_slider.get()
        start_time = float(start_time_entry.get())
        colormap = colormap_var.get()


        # Generate the filename based on the selected ablation time, noise, and start time
        data = read_data_file(IMS_data)

        # Step 2: Preprocess the data
        preprocessed_data = preprocess_data(data)

        # Step 3: Collect data for each ablation time interval
        collected_data = collect_data_by_ablation_time(preprocessed_data, start_time, ablation_time)

        # Step 4: Subtract background noise and calculate intensities
        intensities = subtract_background_noise(collected_data, noise_value)

        # Step 5: Scale and perform logarithmic transformation
        scaled_intensities = scale_and_log_transform(intensities, AbrationTimes)

        # Step 6: Reshape the data for visualization
        reshaped_data = reshape_data_for_visualization(scaled_intensities, Length)

        # Step 7: Reverse alternate rows
        reversed_data = reverse_alternate_rows(reshaped_data)
        im = cv2.imread(microscope_image_file_path)
        cv2.imwrite('GuiCode\\temp\\genIMG.bmp',im)
        
        save_image(reversed_data, 'GuiCode\\temp\\genIMG.bmp',colormap)    
        #from new import overlayMain
        #overlayMain()    
        script_path = "GuiCode\\overlay.py"
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            # Script executed successfully
            print("Script output:")
            print(result.stdout)
        else:
            # Script execution failed
            print("Script failed with error:")
            print(result.stderr)
    
        
        
        
        
        
    button_frame = tk.Frame(root)
    button_frame.pack()

    generate_plot_button = tk.Button(button_frame, text="Refrence IMG Overlay", command=generate_3d_plot)
    generate_plot_button.grid(row=0, column=0, padx=10, pady=10)

    save_image_button = tk.Button(button_frame, text="Save as .bmp", command=save_image_handler)
    save_image_button.grid(row=0, column=1, padx=10, pady=10)

    save_img_button = tk.Button(button_frame, text="Save as Analyze File", command=save_img_handler2)
    save_img_button.grid(row=0, column=2, padx=10, pady=10)
    
    analyze_img_button = tk.Button(button_frame, text="Further Analysis", command=further_analysis)
    analyze_img_button.grid(row=0,column=4,padx=10,pady=10)





    

    # Label for displaying the ablation time, noise, and start time
    label_text = tk.StringVar()
    label = tk.Label(root, textvariable=label_text)
    label.pack(pady=(10))




    # Set the initial slider values and update the image
    noise_slider.set(Noise)
    update_image()

    root.mainloop()


if __name__ == "__main__":
    main()