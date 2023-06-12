import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
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
    IMS_data = '20230528_caffeine_MSI_3.txt'
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

    #No need to save or display images
    # Step 8: Save the mass spectrometry image
    #save_image(reversed_data, 'MSI_image.bmp')

    # Step 9: Display the mass spectrometry image with colorbar
    #display_image_with_colorbar(reversed_data)

    # GUI code
    root = tk.Tk()
    root.title("Mass Spectrometry Image GUI")

    # Load and display the initial image
    image = Image.open('MSI_image.bmp')
    image = image.resize((400, 400), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(image)
    image_label = tk.Label(root, image=img_tk)
    image_label.pack(pady=10)

    def update_image(event=None):
        """
        Update the image based on the slider values.
        """
        ablation_time = float(ablation_entry.get())
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

    # Entry box for entering the ablation time
    ablation_label = tk.Label(root, text="Ablation Time (Mass Alignment):")
    ablation_label.pack(pady=10)
    ablation_entry = tk.Entry(root)
    ablation_entry.insert(0, str(AbrationTime))
    ablation_entry.bind("<Return>", update_image)
    ablation_entry.pack()

    # Slider for adjusting the noise
    noise_label = tk.Label(root, text="Noise Reduction:")
    noise_label.pack(pady=5)
    noise_slider = tk.Scale(root, from_=0, to=100, length=300, orient=tk.HORIZONTAL, resolution=1, command=update_image)
    noise_slider.pack()

    # Entry box for entering the start time
    start_time_label = tk.Label(root, text="Start Time:")
    start_time_label.pack(pady=10)
    start_time_entry = tk.Entry(root)
    start_time_entry.insert(0, str(StartTime))
    start_time_entry.bind("<Return>", update_image)
    start_time_entry.pack(pady=(0,15))

    # Save Image button
    def save_image_handler():
        ablation_time = float(ablation_entry.get())
        noise_value = noise_slider.get()
        start_time = float(start_time_entry.get())

        filename = f"MSI_image_AL-{ablation_time}_N-{noise_value}_ST-{start_time}.bmp"
        save_image(reversed_data, filename)
        print(f"Image saved as {filename}")

    save_image_button = tk.Button(root, text="Save Image", command=save_image_handler)
    save_image_button.pack(pady=(0,10))

    # Label for displaying the ablation time, noise, and start time
    label_text = tk.StringVar()
    label = tk.Label(root, textvariable=label_text)
    label.pack()

    # Set the initial slider values and update the image
    noise_slider.set(Noise)
    update_image()

    root.mainloop()


if __name__ == "__main__":
    main()