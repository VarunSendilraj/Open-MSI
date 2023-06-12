import numpy as np
import matplotlib.pyplot as plt

# Step 1: Import the data
data = np.loadtxt("230111_caffeini_3_195_1.txt")

# Step 2: Convert the data to a format for imaging
x = data[:, 0]  # x-coordinates (time)
y = data[:, 1]  # y-coordinates (intensity)

# Step 3: Perform image processing (optional)
# No specific image processing steps mentioned, skipping this for now

# Step 4: Visualize the data
plt.plot(x, y)
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.title('Mass Spectrometry Data')
plt.show()
