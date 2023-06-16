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


def save_image(reshaped_data, filename):
    """
    Save the mass spectrometry image as an image file.
    """
    image = np.array(reshaped_data)
    plt.imsave(filename, image, cmap="jet")
    
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


def main():
    # File selection dialog to choose the IMS data file
    
    
    root = tk.Tk()
    root.withdraw()
    IMS_data = filedialog.askopenfilename(title="Select IMS Data File")
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
    image = image.resize((400, 400), Image.ANTIALIAS)
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

        # Step 8: Save the mass spectrometry image
        save_image(reversed_data, 'MSI_image.bmp')
        
        

        # Update the image in the GUI
        new_image = Image.open('MSI_image.bmp')
        new_image = new_image.resize((400, 400), Image.ANTIALIAS)
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

        save_image(reversed_data, file_path)
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
        save_image(reversed_data,'USE.bmp')
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
        """
        Visualize the mass spectrometry data in 3D.
        """
        x, y = np.meshgrid(range(len(reversed_data)), range(len(reversed_data[0])))
        z = np.array(reversed_data)

        # Step 9: Visualize the mass spectrometry data in 3D
        data = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = []
        y = []
        z = []

        for point in data:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])

        scatter = ax.scatter(x, y, z, c=z, cmap='jet')
        colorbar = plt.colorbar(scatter, pad=0.05, shrink=0.5, aspect=10)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Intensity')
        
        plt.show()


    button_frame = tk.Frame(root)
    button_frame.pack()

    generate_plot_button = tk.Button(button_frame, text="Generate 3D Plot", command=generate_3d_plot)
    generate_plot_button.grid(row=0, column=0, padx=10, pady=10)

    save_image_button = tk.Button(button_frame, text="Save as .bmp", command=save_image_handler)
    save_image_button.grid(row=0, column=1, padx=10, pady=10)

    save_img_button = tk.Button(button_frame, text="Save as Analyze File", command=save_img_handler2)
    save_img_button.grid(row=0, column=2, padx=10, pady=10)



    

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