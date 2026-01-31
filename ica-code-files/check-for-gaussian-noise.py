import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, normaltest

img = cv2.imread('sample-watermark.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
img = img - np.mean(img)  # ICA assumes zero-mean

pixels = img.flatten()

# Histogram
plt.hist(pixels, bins=100, density=True)
plt.title("Pixel Intensity Distribution")
plt.show()

# Kurtosis
k = kurtosis(pixels, fisher=True)  # fisher=True gives excess kurtosis
print("Excess Kurtosis:", k)

# Normality test
stat, p = normaltest(pixels)
print("p-value for Gaussian test:", p)