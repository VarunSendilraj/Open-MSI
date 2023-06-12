# Time of Flight Mass Spectrometry Data to Image Conversion GUI
This is a Python GUI application that converts Time of Flight Mass Spectrometry (TOF-MS) data into image data. The application allows users to interactively find the optimal mass alignment value, noise value, and start time for the conversion process. The converted image can be saved or displayed with a colorbar.

## Background
Time of Flight Mass Spectrometry is a technique used in analytical chemistry to measure the mass-to-charge ratio of ions. It involves ionizing a sample, accelerating the ions into a flight tube, and measuring the time it takes for each ion to reach a detector. The resulting data is typically represented as a two-dimensional matrix, with the x-axis representing the time of flight and the y-axis representing the mass-to-charge ratio.

Converting TOF-MS data into an image format is useful for visualization and further analysis of the data. It allows researchers to observe patterns and trends in the ion intensity across different mass and time values, which can provide insights into the composition and structure of the analyzed sample.


## Installation
1. Ensure that you have Python installed on your system. This code was developed using Python 3.9, but it should work with other Python 3 versions as well.

2. Install the required dependencies by running the following command:


```
pip install tkinter pillow opencv-python matplotlib
```

3. Download the folder and save it on your computer

## Usage
1. Run the Python script using the following command:

    ```python tofms_gui.py```

2. The application will open a file selection dialog. Choose the TOF-MS data file (.txt extention) that you want to convert.

3. Once the data file is loaded, the GUI will display the initial image. The ablation time, mass alignment value, and noise value sliders will be set to their default positions.

4. Adjust the sliders and entry boxes for the ablation time, mass alignment value, and noise value to fine-tune the conversion parameters. The changes will be reflected in real-time on the displayed image.

5. The ablation time slider controls the length of each interval during data collection. Move the slider to the left or right to decrease or increase the ablation time, respectively. Alternatively, you can manually enter the desired value in the corresponding entry box.

6. The mass alignment value slider determines the mass alignment point for the conversion process. Adjust the slider or enter a value in the entry box to align the mass axis of the image. This helps to align the peaks of interest and improve the accuracy of the conversion.

7. The noise value slider allows you to set the background noise level to subtract from the data. Move the slider or enter a value in the entry box to adjust the noise threshold. Subtracting the background noise enhances the signal-to-noise ratio and improves the clarity of the image.

8. As you make adjustments to the conversion parameters, the image will be updated in real-time to reflect the changes. This allows you to visually assess the effect of parameter modifications on the image quality.

9. To save the converted image, click the "Save Image" button. Choose a file name and location to save the image as a bitmap file (.bmp).

10. If you want to display the image with a colorbar, check the "Display Colorbar" checkbox. The image will be displayed along with a colorbar indicating the intensity scale. This can be helpful for interpreting the ion intensities in the image.

11. To exit the application, click the "Exit" button or close the window.


## Conclusion
The Time of Flight Mass Spectrometry Data to Image Conversion GUI provides an interactive tool for converting TOF-MS data into image format. By adjusting the conversion parameters, users can enhance the visual representation of the data, align the mass axis, and subtract background noise. The converted images can be saved for further analysis or displayed with a colorbar for improved interpretation.