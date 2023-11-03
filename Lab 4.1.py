import matplotlib.pyplot as plt
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
from skimage.filters import threshold_otsu
from scipy.stats import linregress
import glob

time = 0
time_list = []
area_list = []

# Update the path and pattern based on your file structure
# Print the files returned by glob.glob() for debugging
files = glob.glob(r"E:\Usman\Image Processing\Lab4 P\*.*")
print("Files found:", files)

for file in files:
    img = io.imread(file)
    entropy_img = entropy(img, disk(3))
    thresh = threshold_otsu(entropy_img)
    binary = entropy_img <= thresh
    scratch_area = np.sum(binary == 1)
    print("Time =", time, "hours  ", "Scratch Area =", scratch_area, "pixel²")
    time_list.append(time)
    area_list.append(scratch_area)
    time += 1

plt.plot(time_list, area_list, 'bo')  # Print blue dots scatter plot
slope, intercept, r_value, p_value, std_err = linregress(time_list, area_list)
print("y = ", slope, "x", " + ", intercept)
print("R² = ", r_value**2)

plt.show()